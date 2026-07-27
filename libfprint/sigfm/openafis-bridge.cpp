#include "openafis-bridge.hpp"
#include "fpi-log.h"

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
        minutiae.emplace_back(
            OpenAFIS::Minutia::Type::RidgeEnding,
            static_cast<uint16_t>(std::lround(kp.pt.x)),
            static_cast<uint16_t>(std::lround(kp.pt.y)),
            0);
    }
    return minutiae;
}

static void estimate_dimensions(
    const std::vector<OpenAFIS::Minutia>& minutiae,
    uint16_t& width, uint16_t& height)
{
    if (width > 0 && height > 0)
        return;

    uint16_t max_x = 0, max_y = 0;
    for (const auto& m : minutiae) {
        if (m.x() > max_x) max_x = m.x();
        if (m.y() > max_y) max_y = m.y();
    }
    if (max_x > 0) width = max_x + 1;
    if (max_y > 0) height = max_y + 1;
}

OpenAFIS::Fingerprint build_fingerprint_from_minutiae(
    const std::vector<OpenAFIS::Minutia>& minutiae,
    uint16_t img_width, uint16_t img_height)
{
    estimate_dimensions(minutiae, img_width, img_height);

    std::vector<std::vector<OpenAFIS::Minutia>> fps;
    fps.push_back(minutiae);

    SiftTemplate t(0);
    if (!t.build({img_width, img_height}, fps)) {
        return OpenAFIS::Fingerprint(0, 0);
    }
    return t.fingerprint();
}

int match_score(const SigfmImgInfo* probe, const SigfmImgInfo* candidate)
{
    auto probe_minutiae = keypoints_to_minutiae(probe, 50);
    auto candidate_minutiae = keypoints_to_minutiae(candidate, 50);

    fp_dbg("openafis: probe keypts=%d minu=%zu, candidate keypts=%d minu=%zu, dims=(%u,%u)/(%u,%u)",
        (int)probe->keypoints.size(), probe_minutiae.size(),
        (int)candidate->keypoints.size(), candidate_minutiae.size(),
        probe->width, probe->height, candidate->width, candidate->height);

    if (probe_minutiae.size() < 2 || candidate_minutiae.size() < 2) {
        fp_dbg("openafis: too few minutiae, returning 0");
        return 0;
    }

    auto probe_fp = build_fingerprint_from_minutiae(
        probe_minutiae, probe->width, probe->height);
    auto candidate_fp = build_fingerprint_from_minutiae(
        candidate_minutiae, candidate->width, candidate->height);

    fp_dbg("openafis: fp minuCount=%zu/%zu triplets=%zu/%zu",
        probe_fp.minutiaeCount(), candidate_fp.minutiaeCount(),
        probe_fp.triplets().size(), candidate_fp.triplets().size());

    if (probe_fp.minutiaeCount() < 2 || candidate_fp.minutiaeCount() < 2) {
        fp_dbg("openafis: fp too few minutiae, returning 0");
        return 0;
    }

    OpenAFIS::MatchSimilarity match;
    OpenAFIS::Param param;
    param.MaximumLocalDistance = 50;
    param.MaximumGlobalDistance = 50;
    param.MinimumMinutiae = 2;
    uint8_t score = 0;
    match.compute(score, probe_fp, candidate_fp, param);

    fp_dbg("openafis: raw score=%d", (int)score);
    return static_cast<int>(score);
}

} // namespace sigfm_openafis
