
sigfm_sources = ['sigfm.cpp', 'openafis-bridge.cpp']

opencv = dependency('opencv4', required: true)
doctest = dependency('doctest', required: true)
openafis_dep = dependency('openafis',
    fallback: ['openafis', 'openafis_dep'])

inc_dir = include_directories('..')

libsigfm = static_library('sigfm',
        sigfm_sources,
        dependencies: [opencv, glib_dep, openafis_dep],
        include_directories: inc_dir
)
sigfm_tests = executable('sigfm-tests', ['./tests.cpp'], dependencies: [doctest, opencv], link_with: [libsigfm])
