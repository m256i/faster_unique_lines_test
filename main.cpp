// Type your code here, or load an example.
#include <array>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <fstream>

#include <stdint.h>
#include <highwayhash/highwayhash.h>

#ifdef _WIN32
#include "file_buffer.h"
#else
#include "common.h"
#endif

__attribute__((const)) inline std::uint32_t
fast_getline(const char *_str)
{
  unsigned long long index   = 0;
  const __m256i all_newlines = _mm256_set1_epi8('\n');
  __m256i hits               = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i *)(_str + index)), all_newlines);
  std::uint32_t outmask      = _mm256_movemask_epi8(hits);
  while (outmask == 0) [[likely]]
  {
    index += sizeof(__m256i);
    hits    = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i *)(_str + index)), all_newlines);
    outmask = _mm256_movemask_epi8(hits);
  }
  return __builtin_ctz(outmask) + index;
}

// TODO: can be AVX2
__attribute__((const)) inline bool
strcmp_(const char *__restrict__ a, const char *__restrict__ b, unsigned long long len)
{
  unsigned long long c = 0;
  while (a[c] == b[c] && (a[c] != '\n' && b[c] != '\n'))
  {
    c++;
  }
  return c == len;
}

__attribute__((const)) inline bool
test_nonz(__m256i v)
{
  __m256i vcmp = _mm256_cmpeq_epi32(v, _mm256_setzero_si256());
  return _mm256_testz_si256(vcmp, vcmp);
}

__attribute__((const)) usize
npow2(usize _num)
{
  usize lzc = _lzcnt_u64(_num);
  return ((_num << 1) & (0xffffffffffffffff << (64 - lzc)));
}

void
print256epi32(__m256i _vec)
{
  puts("wat");
  alignas(32) u64 arr[4]{};
  std::memcpy(arr, &_vec, sizeof(__m256i));

  std::cout << "vector: \n";
  for (const auto ite : arr)
  {
    std::cout << "    " << ite << "\n";
  }
}

int
main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  // NOTE: (IMPORTANT): this version currently lacks collisions lol

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  usize counter = 0;
  std::string ipt_buf;

#ifdef _WIN32

  auto fsize = xenon_file::get_fsize(argv[1]);
  ipt_buf.reserve(fsize);
  xenon_file::read_file(argv[1], ipt_buf);

#else

  auto stream = std::ifstream(argv[1], std::ios::binary);
  auto buf    = std::stringstream();

  stream.seekg(0, std::ios::end);
  size_t fsize = stream.tellg();
  std::string buffer(fsize, ' ');
  stream.seekg(0);
  stream.read(&buffer[0], fsize);
  ipt_buf = buffer;

#endif

  const auto mapsize = (usize)(npow2(fsize / 45));

  HH_ALIGNAS(32) const highwayhash::HHKey key = {1, 1, 1, 1};
  highwayhash::HHStateT<HH_TARGET_AVX2> state(key);

  std::vector<uptr> hashmap(mapsize, 0);
  u64 current_index = 0, next_index = 0;

#ifdef _WIN32
  SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif

  alignas(32) std::array<u64, 4> hash_indeces;
  std::array<uptr, 4> ptr_buf;
  u64 result_ = 0;

  auto t1 = high_resolution_clock::now();

  while (current_index != fsize) [[likely]]
  {
    for (usize j = 0; j != 4 && (current_index != fsize); ++j)
    {
      next_index = current_index + fast_getline((const char *)((uptr)ipt_buf.data() + current_index));
      highwayhash::HighwayHashT(&state, ipt_buf.data() + current_index, next_index - current_index, &result_);
      ptr_buf[j]      = (uptr)ipt_buf.data() + current_index;
      hash_indeces[j] = result_ % mapsize;
      current_index   = next_index + 1;
    }

    __m256i hash_index_vector = _mm256_load_si256((__m256i *)&hash_indeces);
    __m256i map_entries       = _mm256_permute4x64_epi64(_mm256_i64gather_epi64((i64 *)hashmap.data(), hash_index_vector, 8), 0b00011011);

    while (test_nonz(map_entries)) [[unlikely]] // hashmap[hashvalue] != 0 equivalent
    {
      const __m256i ones           = _mm256_set1_epi32(1);
      const __m256i extended_lanes = _mm256_cmpgt_epi32(map_entries, _mm256_set1_epi32(0));

      /* + 1 % mapsize */
      hash_index_vector = _mm256_blendv_epi8(hash_index_vector, _mm256_add_epi64(ones, hash_index_vector), _mm256_cmpgt_epi64(map_entries, _mm256_set1_epi32(0)));
      hash_index_vector = _mm256_and_si256(hash_index_vector, _mm256_set1_epi64x(mapsize - 1));
      map_entries = _mm256_permute4x64_epi64(_mm256_mask_i64gather_epi64(_mm256_set1_epi32(0), (i64 *)hashmap.data(), hash_index_vector, extended_lanes, sizeof(uptr)), 0b00011011);
    }

    _mm256_storeu_si256((__m256i *)hash_indeces.data(), hash_index_vector);

    for (usize k = 0; k != 4; ++k)
    {
      ++counter;
      hashmap[hash_indeces[k]] = ptr_buf[k];
    }
  }

  auto t2 = high_resolution_clock::now();
  /* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;

  std::cout << "unique lines: " << counter << "  " << (ms_double.count() / 1000.f) << "s\n";
}