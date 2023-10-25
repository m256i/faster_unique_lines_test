project(
    'proj',
    ['cpp'],
    version: '0.1',
    default_options: [
        'warning_level=3',
        'cpp_std=c++20',
        'backend=ninja'
    ]
)
sources = run_command('py', 'globber.py', check : true).stdout().strip().split('\n')
 
message('source files: ')
foreach f : sources
  message('   ' + f)
endforeach

cmake             = import('cmake')

root_path = meson.global_source_root()
include_path = root_path / 'include'
hh_path = root_path / 'highwayhash'

executable(
  'fast_unique',
  [sources],
  cpp_args  : ['-march=znver2', '-O3', '-fno-exceptions', '-fno-rtti', '-g'],
  include_directories : [include_path, hh_path],
  dependencies : [],
  install : true
)

# meson build --wipe -Ddebug=true -Doptimization=3