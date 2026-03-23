// eval_sigfm_1n.cpp
// Evaluation tool for 1:N fingerprint matching using libsigfm

#include <string>
#include <vector>
#include <map>
#include <random>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <future>

#include <glib.h>
#include <glib/gprintf.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include <sigfm/sigfm.h>

namespace fs = std::filesystem;

// === Device Simulator ===
class DeviceSimulator {
public:
    virtual cv::Mat simulate(const cv::Mat& image) = 0;
    virtual ~DeviceSimulator() = default;
};

class CS9711Simulator : public DeviceSimulator {
private:
    cv::Size target_size{68, 118};
    std::pair<int, int> angle_range{-180, 180};
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    std::mt19937 rng;

public:
    explicit CS9711Simulator(int seed) : rng(seed) {}

    cv::Mat simulate(const cv::Mat& image) override {
        cv::Mat img = image.clone();
        std::uniform_int_distribution<int> angle_dist(angle_range.first, angle_range.second);
        std::uniform_int_distribution<int> x_dist(0, 1000000), y_dist(0, 1000000);

        for (int i = 0; i < 10; ++i) {
            int angle = angle_dist(rng);
            cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
            cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
            cv::Mat rotated;
            cv::warpAffine(img, rotated, rot_mat, img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));

            cv::Mat thresh;
            cv::threshold(rotated, thresh, 220, 255, cv::THRESH_BINARY_INV);

            cv::Mat eroded;
            cv::erode(thresh, eroded, erode_kernel);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(eroded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            std::vector<cv::Point> all_points;
            for (const auto& cnt : contours) {
                all_points.insert(all_points.end(), cnt.begin(), cnt.end());
            }

            if (all_points.empty()) {
                continue;
            }

            cv::Rect bounding_box = cv::boundingRect(all_points);

            if (bounding_box.width < target_size.width || bounding_box.height < target_size.height) {
                continue;
            }

            int x = bounding_box.x + std::uniform_int_distribution<int>(0, bounding_box.width - target_size.width)(rng);
            int y = bounding_box.y + std::uniform_int_distribution<int>(0, bounding_box.height - target_size.height)(rng);

            cv::Mat part = rotated(cv::Rect(x, y, target_size.width, target_size.height)).clone();
            clahe->apply(part, part);
            return part;
       }
        return cv::Mat();
    }
};

class DummySimulator : public DeviceSimulator {
public:
    cv::Mat simulate(const cv::Mat& image) override {
        return image.clone();
    }
};

// === Feature Data Wrapper ===
struct FeatureData {
    SigfmImgInfo* info = nullptr;

    ~FeatureData() {
        if (info) sigfm_free_info(info);
    }

    FeatureData() = default;
    FeatureData(const FeatureData&) = delete;
    FeatureData& operator=(const FeatureData&) = delete;

    FeatureData(FeatureData&& other) noexcept : info(other.info) {
        other.info = nullptr;
    }

    FeatureData(SigfmImgInfo* _info) : info(sigfm_copy_info(_info)) {}

    FeatureData& operator=(FeatureData&& other) noexcept {
        if (this != &other) {
            if (info) sigfm_free_info(info);
            info = other.info;
            other.info = nullptr;
        }
        return *this;
    }
};

std::vector<fs::path> get_image_files(const fs::path& dir);
std::string extract_user_finger(const fs::path& p);

// === Utility Functions ===
std::vector<fs::path> get_image_files(const fs::path& dir) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        const auto& path = entry.path();
        if (entry.is_regular_file()) {
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".tif" || ext == ".tiff" || ext == ".jpg" || ext == ".jpeg") {
                files.push_back(path);
            }
        }
    }
    return files;
}

std::string extract_user_finger(const fs::path& p) {
    return p.stem().string();  // Use full stem as user+finger ID
}

// === Main Evaluation Logic ===
int main(int argc, char* argv[]) {
    fs::path input_dir;
    fs::path output_file;
    std::string device_type = "CS9711";

    gchar* input_dir_cstr = nullptr;
    gchar* output_file_cstr = nullptr;
    gchar* device_type_cstr = g_strdup(device_type.c_str());
    gint num_crops = 10;
    gint gallery_size = 15;
    gint n_samples = 50000;
    gint seed = 12345;

    GError* error = nullptr;

    GOptionContext* context = g_option_context_new("- SIGFM 1:N Evaluation Tool");

    GOptionEntry entries[] = {
        { "input", 'i', 0, G_OPTION_ARG_FILENAME, &input_dir_cstr,
          "Path to input images", "PATH" },
        { "output", 'o', 0, G_OPTION_ARG_FILENAME, &output_file_cstr,
          "Path to output CSV file", "PATH" },
        { "device", 0, 0, G_OPTION_ARG_STRING, &device_type_cstr,
          "Device type [CS9711|dummy]", "TYPE" },
        { "crops", 'p', 0, G_OPTION_ARG_INT, &num_crops,
          "Number of source crops (0 = full image)", "N" },
        { "gallery-size", 'g', 0, G_OPTION_ARG_INT, &gallery_size,
          "Number of enrolled crops per user", "N" },
        { "n-samples", 't', 0, G_OPTION_ARG_INT, &n_samples,
          "Number of random probe/gallery pairs", "N" },
        { "seed", 0, 0, G_OPTION_ARG_INT, &seed,
          "Random seed", "SEED" },
        { nullptr }
    };

    g_option_context_add_main_entries(context, entries, nullptr);

    if (!g_option_context_parse(context, &argc, &argv, &error)) {
        g_fprintf(stderr, "Option parsing failed: %s\n", error->message);
        g_error_free(error);
        g_option_context_free(context);
        return 1;
    }

    // Copy to Args
    if (input_dir_cstr) {
        input_dir = fs::path(input_dir_cstr);
    } else {
        g_fprintf(stderr, "Error: --input is required.\n");
        g_option_context_free(context);
        return 1;
    }

    if (output_file_cstr) {
        output_file = fs::path(output_file_cstr).replace_extension(".csv");
    } else {
        g_fprintf(stderr, "Error: --output is required.\n");
        g_option_context_free(context);
        return 1;
    }

    // Free GLib-allocated strings
    g_free(input_dir_cstr);
    g_free(output_file_cstr);
    g_free(device_type_cstr);
    g_option_context_free(context);

    std::cout << "Parsed args:\n";
    std::cout << "  Input: " << input_dir << "\n";
    std::cout << "  Output: " << output_file << "\n";
    std::cout << "  Device: " << device_type << "\n";
    std::cout << "  Crops: " << num_crops << "\n";
    std::cout << "  Gallery size: " << gallery_size << "\n";
    std::cout << "  N samples: " << n_samples << "\n";
    std::cout << "  Seed: " << seed << "\n";

    std::mt19937 gen(seed);
    cv::setRNGSeed(seed);

    std::unique_ptr<DeviceSimulator> simulator;
    if (device_type == "CS9711") {
        simulator = std::make_unique<CS9711Simulator>(seed);
    } else {
        simulator = std::make_unique<DummySimulator>();
    }

    // Collect files
    auto files = get_image_files(input_dir);
    std::shuffle(files.begin(), files.end(), gen);

    // Maps: user -> list of enrolled templates
    std::map<std::string, std::vector<FeatureData>> enrolls;
    std::map<std::pair<std::string, int>, FeatureData> probes;  // (user, idx)

    std::cout << "Extracting features...\n";

    for (const auto& f : files) {
        cv::Mat image = cv::imread(f.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;

        std::string user_finger = extract_user_finger(f);

        if (num_crops == 0) {
            // Full image
            auto* info = sigfm_extract(image.data, image.cols, image.rows);
            if (info) {
                probes[{user_finger, 0}] = FeatureData{info};
                enrolls[user_finger].push_back(FeatureData{info});
                sigfm_free_info(info);
            }
        } else {
            // Crop-based
            for (int i = 0; i < num_crops; ++i) {
                cv::Mat crop = simulator->simulate(image);
                if (!crop.empty()) {
                    auto* info = sigfm_extract(crop.data, crop.cols, crop.rows);
                    if (info) {
                        probes[{user_finger, i}] = FeatureData{info};
                        sigfm_free_info(info);
                    }
                }
            }
            for (int i = 0; i < gallery_size; ++i) {
                cv::Mat crop = simulator->simulate(image);
                if (!crop.empty()) {
                    auto* info = sigfm_extract(crop.data, crop.cols, crop.rows);
                    if (info) {
                        enrolls[user_finger].push_back(FeatureData{info});
                        sigfm_free_info(info);
                    }
                }
            }
        }
    }

    std::cout << "Matching 1:N...\n";

    // Generate random pairs
    std::vector<std::pair<std::string, int>> probe_keys;
    std::vector<std::string> enroll_keys;
    for (const auto& [k, _] : probes) probe_keys.push_back(k);
    for (const auto& [k, _] : enrolls) enroll_keys.push_back(k);

    size_t max_pairs = probe_keys.size() * enroll_keys.size();
    size_t num_to_process = std::min(static_cast<size_t>(n_samples), max_pairs);

    std::vector<int> indices(max_pairs);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    indices.resize(num_to_process);

    // match in parallel threads

    // Match using std::async
    std::vector<std::pair<bool, int>> matches;
    matches.reserve(num_to_process);

    int num_threads = std::max(1, std::min(
        static_cast<int>(std::thread::hardware_concurrency()),
        static_cast<int>(num_to_process)
    ));

    std::vector<std::future<std::vector<std::pair<bool, int>>>> futures;

    int batch_size = (num_to_process + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * batch_size;
        size_t end = std::min(start + batch_size, num_to_process);

        if (start >= num_to_process) continue;

        // Capture by value where needed, reference to shared data as const&
        futures.emplace_back(std::async(std::launch::async, [=, &probe_keys, &enroll_keys, &probes, &enrolls]() {
            std::vector<std::pair<bool, int>> local_matches;
            local_matches.reserve(end - start);

            for (size_t i = start; i < end; ++i) {
                int idx = indices[i];
                auto& pkey = probe_keys[idx % probe_keys.size()];
                auto& ekey = enroll_keys[idx / probe_keys.size()];

                bool is_genuine = (pkey.first == ekey);

                int max_score = 0;
                for (const auto& enrolled_feat : enrolls.at(ekey)) {
                    int score = sigfm_match_score(probes.at(pkey).info, enrolled_feat.info);
                    if (score > max_score) max_score = score;
                }

                local_matches.emplace_back(is_genuine, max_score);
            }

            return local_matches;
        }));
    }

    // Собираем результаты
    for (auto& fut : futures) {
        auto batch = fut.get();  // Блокируется до завершения
        matches.insert(matches.end(), batch.begin(), batch.end());
    }

    // Write CSV
    std::ofstream out(output_file);
    out << "is_genuine,score\n";
    for (const auto& [genuine, score] : matches) {
        out << (genuine ? "1" : "0") << "," << score << "\n";
    }
    out.close();

    std::cout << "Evaluation complete. Results saved to " << output_file << "\n";
    return 0;
}
