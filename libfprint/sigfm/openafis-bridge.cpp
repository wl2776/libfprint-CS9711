#include "openafis-bridge.hpp"
#include "fpi-log.h"

#include <Dimensions.h>
#include <FastMath.h>
#include <Field.h>
#include <Match.h>
#include <MinutiaPoint.h>
#include <Param.h>
#include <Template.h>

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sigfm_openafis {

namespace {

class SiftTemplate : public OpenAFIS::Template<uint32_t, OpenAFIS::Fingerprint> {
public:
    explicit SiftTemplate(uint32_t id)
        : Template(id) {}

    bool build(const OpenAFIS::Dimensions& dims,
               const std::vector<std::vector<OpenAFIS::Minutia>>& fps)
    {
        return this->load(dims, fps);
    }

    const OpenAFIS::Fingerprint& fingerprint() const
    {
        return this->fingerprints()[0];
    }

    bool has_fingerprint() const
    {
        return !this->fingerprints().empty();
    }
};

OpenAFIS::Fingerprint build_fingerprint_from_minutiae(
    const std::vector<OpenAFIS::Minutia>& minutiae,
    uint16_t img_width, uint16_t img_height)
{
    if (minutiae.size() < 3)
        return OpenAFIS::Fingerprint(0, 0);

    if (img_width == 0 || img_height == 0) {
        uint16_t max_x = 0, max_y = 0;
        for (const auto& m : minutiae) {
            if (m.x() > max_x) max_x = m.x();
            if (m.y() > max_y) max_y = m.y();
        }
        if (max_x > 0) img_width = max_x + 1;
        if (max_y > 0) img_height = max_y + 1;
    }

    if (img_width == 0 || img_height == 0)
        return OpenAFIS::Fingerprint(0, 0);

    std::vector<std::vector<OpenAFIS::Minutia>> fps;
    fps.push_back(minutiae);

    SiftTemplate t(0);
    if (!t.build({img_width, img_height}, fps) || !t.has_fingerprint()) {
        return OpenAFIS::Fingerprint(0, 0);
    }
    return t.fingerprint();
}

static std::vector<cv::KeyPoint> filter_by_response(
    const std::vector<cv::KeyPoint>& kps, int max_count)
{
    std::vector<cv::KeyPoint> sorted(kps.begin(), kps.end());
    std::sort(sorted.begin(), sorted.end(),
        [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
            return a.response > b.response;
        });
    if ((int)sorted.size() > max_count)
        sorted.resize((size_t)max_count);
    return sorted;
}

} // namespace

int match_score(const SigfmImgInfo* probe, const SigfmImgInfo* candidate)
{
    if (!probe || !candidate) {
        fp_dbg("openafis: null pointer");
        return 0;
    }
    if (probe->keypoints.empty() || candidate->keypoints.empty()) {
        fp_dbg("openafis: no keypoints");
        return 0;
    }
    if (probe->descriptors.empty() || candidate->descriptors.empty()) {
        fp_dbg("openafis: empty descriptors");
        return 0;
    }

    std::vector<std::vector<cv::DMatch>> knn_matches;
    auto bfm = cv::BFMatcher::create(cv::NORM_L2);
    bfm->knnMatch(probe->descriptors, candidate->descriptors, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m.size() < 2) continue;
        if (m[0].distance < 0.8f * m[1].distance) {
            if (m[0].queryIdx < (int)probe->keypoints.size() &&
                m[0].trainIdx < (int)candidate->keypoints.size())
                good_matches.push_back(m[0]);
        }
    }
    std::sort(good_matches.begin(), good_matches.end(),
        [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

    int match_count = (int)good_matches.size();
    fp_dbg("openafis: descriptor matches=%d (probe kpts=%d, cand kpts=%d)",
        match_count, (int)probe->keypoints.size(), (int)candidate->keypoints.size());

    if (match_count < 5) {
        fp_dbg("openafis: too few descriptor matches");
        return 0;
    }

    int n = std::min(match_count, 50);
    std::vector<OpenAFIS::Minutia> probe_minu, candidate_minu;
    probe_minu.reserve(n);
    candidate_minu.reserve(n);
    for (int i = 0; i < n; i++) {
        const auto& pk = probe->keypoints[good_matches[i].queryIdx];
        const auto& ck = candidate->keypoints[good_matches[i].trainIdx];
        probe_minu.emplace_back(OpenAFIS::Minutia::Type::RidgeEnding,
            (uint16_t)std::lround(pk.pt.x), (uint16_t)std::lround(pk.pt.y), 0);
        candidate_minu.emplace_back(OpenAFIS::Minutia::Type::RidgeEnding,
            (uint16_t)std::lround(ck.pt.x), (uint16_t)std::lround(ck.pt.y), 0);
    }

    auto deduplicate = [](std::vector<OpenAFIS::Minutia>& minu) {
        std::sort(minu.begin(), minu.end(),
            [](const OpenAFIS::Minutia& a, const OpenAFIS::Minutia& b) {
                if (a.x() != b.x()) return a.x() < b.x();
                return a.y() < b.y();
            });
        minu.erase(std::unique(minu.begin(), minu.end(),
            [](const OpenAFIS::Minutia& a, const OpenAFIS::Minutia& b) {
                return a.x() == b.x() && a.y() == b.y();
            }), minu.end());
    };
    deduplicate(probe_minu);
    deduplicate(candidate_minu);

    auto probe_fp = build_fingerprint_from_minutiae(probe_minu, probe->width, probe->height);
    auto candidate_fp = build_fingerprint_from_minutiae(candidate_minu, candidate->width, candidate->height);

    fp_dbg("openafis: fp after filter: minu=%zu/%zu triplets=%zu/%zu",
        probe_fp.minutiaeCount(), candidate_fp.minutiaeCount(),
        probe_fp.triplets().size(), candidate_fp.triplets().size());

    if (probe_fp.minutiaeCount() < 3 || candidate_fp.minutiaeCount() < 3) {
        fp_dbg("openafis: too few fp minutiae for triangulation");
        return 0;
    }

    OpenAFIS::MatchSimilarity match;
    OpenAFIS::Param param;
    param.MaximumLocalDistance = 50;
    param.MaximumGlobalDistance = 50;
    param.MinimumMinutiae = 2;
    uint8_t geo_score = 0;
    match.compute(geo_score, probe_fp, candidate_fp, param);

    fp_dbg("openafis: geo_score=%d, match_count=%d, combined=%d",
        (int)geo_score, match_count, (geo_score >= 5 ? match_count : 0));

    if (geo_score < 5) return 0;
    return match_count;
}

} // namespace sigfm_openafis
