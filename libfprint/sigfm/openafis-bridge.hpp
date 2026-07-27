#pragma once

#include "img-info.hpp"

namespace sigfm_openafis {

int match_score(const SigfmImgInfo* probe, const SigfmImgInfo* candidate);

} // namespace sigfm_openafis
