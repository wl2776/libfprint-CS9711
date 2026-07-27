#pragma once

#include "img-info.hpp"

#include <Fingerprint.h>
#include <Minutia.h>

#include <cstdint>
#include <vector>

namespace sigfm_openafis {

std::vector<OpenAFIS::Minutia> keypoints_to_minutiae(
    const SigfmImgInfo* info, int max_minutiae = 128);

OpenAFIS::Fingerprint build_fingerprint_from_minutiae(
    const std::vector<OpenAFIS::Minutia>& minutiae,
    uint16_t img_width, uint16_t img_height);

int match_score(const SigfmImgInfo* probe, const SigfmImgInfo* candidate);

} // namespace sigfm_openafis
