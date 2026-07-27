#include "openafis-bridge.hpp"

#include <Dimensions.h>
#include <FastMath.h>
#include <Field.h>
#include <Match.h>
#include <MinutiaPoint.h>
#include <Param.h>
#include <Template.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sigfm_openafis {

namespace {
class SiftTemplate : public OpenAFIS::Template<uint32_t, OpenAFIS::Fingerprint> {
public:
    using Template::Template;
    using Template::load;
};
} // namespace

std::vector<OpenAFIS::Minutia> keypoints_to_minutiae(
    const SigfmImgInfo* info, int max_minutiae)
{
    std::vector<cv::KeyPoint> sorted(info->keypoints.begin(), info->keypoints.end());
    std::sort(sorted.begin(), sorted.end(),
        [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
            return a.response > b.response;
        });

    if (static_cast<int>(sorted.size()) > max_minutiae) {
        sorted.resize(static_cast<size_t>(max_minutiae));
    }

    std::vector<OpenAFIS::Minutia> minutiae;
    minutiae.reserve(sorted.size());
    for (const auto& kp : sorted) {
        uint16_t angle = 0;
        if (kp.angle >= 0.0f) {
            angle = static_cast<uint16_t>(std::lround(kp.angle)) % 360;
        }
        minutiae.emplace_back(
            OpenAFIS::Minutia::Type::RidgeEnding,
            static_cast<uint16_t>(std::lround(kp.pt.x)),
            static_cast<uint16_t>(std::lround(kp.pt.y)),
            angle);
    }
    return minutiae;
}

OpenAFIS::Fingerprint build_fingerprint_from_minutiae(
    const std::vector<OpenAFIS::Minutia>& minutiae,
    uint16_t img_width, uint16_t img_height)
{
    std::vector<std::vector<OpenAFIS::Minutia>> fps;
    fps.push_back(minutiae);

    OpenAFIS::Dimensions dims{img_width, img_height};
    SiftTemplate t(0);
    if (!t.load(dims, fps)) {
        return OpenAFIS::Fingerprint(0, 0);
    }
    return t.fingerprints()[0];
}

int match_score(const SigfmImgInfo* probe, const SigfmImgInfo* candidate)
{
    auto probe_minutiae = keypoints_to_minutiae(probe, 128);
    auto candidate_minutiae = keypoints_to_minutiae(candidate, 128);

    if (probe_minutiae.size() < 2 || candidate_minutiae.size() < 2) {
        return 0;
    }

    auto probe_fp = build_fingerprint_from_minutiae(
        probe_minutiae, probe->width, probe->height);
    auto candidate_fp = build_fingerprint_from_minutiae(
        candidate_minutiae, candidate->width, candidate->height);

    if (probe_fp.minutiaeCount() < 2 || candidate_fp.minutiaeCount() < 2) {
        return 0;
    }

    OpenAFIS::MatchSimilarity match;
    OpenAFIS::Param param;
    uint8_t score = 0;
    match.compute(score, probe_fp, candidate_fp, param);
    return static_cast<int>(score);
}

} // namespace sigfm_openafis
