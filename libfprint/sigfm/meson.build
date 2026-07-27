
sigfm_sources = ['sigfm.cpp']

opencv = dependency('opencv4', required: true)
doctest = dependency('doctest', required: true)

inc_dir = include_directories('..')

libsigfm = static_library('sigfm',
        sigfm_sources,
        dependencies: [opencv, glib_dep],
        include_directories: inc_dir
)
sigfm_tests = executable('sigfm-tests', ['./tests.cpp'], dependencies: [doctest, opencv], link_with: [libsigfm])
