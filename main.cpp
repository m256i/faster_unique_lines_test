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

constexpr unsigned int uint_all = ~(unsigned int)0;

__attribute__((const)) inline bool
strcmp_(const char *__restrict__ a, const char *__restrict__ b, unsigned long long len)
{
  unsigned long long c = 0;
  while (c < len)
  {
    auto adata = _mm256_loadu_si256((__m256i const *)&a[c]);
    auto bdata = _mm256_loadu_si256((__m256i const *)&b[c]);

    auto newlines = _mm256_set1_epi8('\n');

    unsigned int are_newlines = _mm256_movemask_epi8(_mm256_cmpeq_epi8(bdata, newlines));
    unsigned int are_equal    = _mm256_movemask_epi8(_mm256_cmpeq_epi8(adata, bdata));

    c += 32;
    if (are_newlines == 0)
    {
      if (are_equal == uint_all) continue;
      else
        break;
    }

    // Index of first newline
    auto zeroes = __builtin_ctz(are_newlines);

    unsigned int should_set = uint_all >> (32 - zeroes - 1);
    return (are_equal & should_set) == should_set;
  }
  return false;
}

__attribute__((const)) inline bool
test_nonz(__m256i v)
{
  return !_mm256_testz_si256(v, v);
}

__attribute__((const)) usize
npow2(usize _num)
{
  usize lzc = _lzcnt_u64(_num);
  return ((_num << 1) & (0xffffffffffffffff << (64 - lzc)));
}

void
print256epi64(__m256i _vec)
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
  std::cout << "fsize " << fsize << std::endl;
  std::string buffer(fsize, ' ');
  stream.seekg(0);
  stream.read(&buffer[0], fsize);
  ipt_buf = buffer;

#endif

  const auto mapsize = (usize)(npow2(fsize));

  HH_ALIGNAS(32) const highwayhash::HHKey key = {1, 1, 1, 1};
  highwayhash::HHStateT<HH_TARGET_AVX2> state(key);

  std::vector<uptr> hashmap(mapsize, 0);
  u64 current_index = 0, next_index = 0;

#ifdef _WIN32
  SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif

  alignas(32) std::array<u64, 4> hash_indeces;
  std::array<u64, 4> sizes;
  std::array<uptr, 4> ptr_buf;
  u64 result_          = 0;
  u64 collisioncounter = 0;

  auto t1 = high_resolution_clock::now();

  while (current_index != fsize) [[likely]]
  {
    for (usize j = 0; j != 4 && (current_index != fsize); ++j)
    {
      next_index = current_index + fast_getline((const char *)((uptr)(ipt_buf.data() + current_index)));
      highwayhash::HighwayHashT(&state, ipt_buf.data() + current_index, next_index - current_index, &result_);
      state.Reset(key);

      // potentially avx too?
      ptr_buf[j] = (uptr)(ipt_buf.data() + current_index);
      sizes[j]   = next_index - current_index;
      // make avx
      hash_indeces[j] = result_ % mapsize;
      current_index   = next_index + 1;
    }

    __m256i hash_index_vector = _mm256_load_si256((__m256i *)hash_indeces.data());
    __m256i map_entries = _mm256_i64gather_epi64((long long *)hashmap.data(), hash_index_vector, 8);

    while (test_nonz(map_entries)) [[unlikely]] // hashmap[hashvalue] != 0 equivalent
    {
      __m256i are_nonzero = _mm256_xor_si256(_mm256_cmpeq_epi64(map_entries, _mm256_setzero_si256()), _mm256_set1_epi64x(~u64(0)));

      auto mask = _mm256_movemask_epi8(are_nonzero);
      for (int i = 0; i < 4; ++i)
      {
        if ((mask & (0b1 << (i * 8))) != 0)
        {
          if (strcmp_((char *)ptr_buf[i], (char *)map_entries[i], sizes[i]))
          {
            are_nonzero[i] = 0;
            --counter;
          }
          else
          {
            ++collisioncounter;
          }
        }
      }

      /* + 1 % mapsize */
      hash_index_vector = _mm256_blendv_epi8(hash_index_vector, _mm256_add_epi64(_mm256_set1_epi64x(1), hash_index_vector), are_nonzero);
      hash_index_vector = _mm256_and_si256(hash_index_vector, _mm256_set1_epi64x(mapsize - 1));

      // if it wasn't 0 then load again, otherwise just 0
      map_entries = _mm256_mask_i64gather_epi64(_mm256_setzero_si256(), (long long *)hashmap.data(), hash_index_vector, are_nonzero, 8);
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
  std::cout << "collisions " << collisioncounter << "\n";
}
