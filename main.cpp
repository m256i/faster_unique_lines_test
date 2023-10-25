// Type your code here, or load an example.
#include <array>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>
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

uint64_t
rdtsc()
{
  return __rdtsc();
}

const std::array<char, sizeof(__m256i)> newlines = {
    '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n',
    '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n'};

std::uint32_t
fast_getline(const char *_str)
{
  unsigned long long index   = 0;
  const __m256i all_newlines = _mm256_loadu_si256((__m256i *)newlines.data());
  __m256i hits = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i *)(_str + index)), all_newlines);
  std::uint32_t outmask = _mm256_movemask_epi8(hits);
  while (outmask == 0)
  {
    index += sizeof(__m256i);
    hits    = _mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i *)(_str + index)), all_newlines);
    outmask = _mm256_movemask_epi8(hits);
  }
  return __builtin_ctz(outmask) + index;
}

// TODO: can be AVX2
bool
strcmp_(const char *__restrict__ a, const char *__restrict__ b, unsigned long long len)
{
  unsigned long long c = 0;
  while (a[c] == b[c] && (a[c] != '\n' && b[c] != '\n'))
  {
    c++;
  }
  return c == len;
}

int
main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  usize counter = 0;
  std::string ipt_buf;

#ifdef _WIN32

  auto fsize = xenon_file::get_fsize(TEXT("D:/git/faster_unique/invalid_tokens.txt"));
  ipt_buf.reserve(fsize);
  xenon_file::read_file(TEXT("D:/git/faster_unique/invalid_tokens.txt"), ipt_buf);

#else

  auto stream = std::ifstream(argv[1], std::ios::binary);
  auto buf    = std::stringstream();

  stream.seekg(0, std::ios::end);
  size_t fsize = stream.tellg();
  std::string buffer(size, ' ');
  stream.seekg(0);
  stream.read(&buffer[0], size);
  ipt_buf = buffer;

#endif

  highwayhash::HHResult64 result;

  const auto mapsize = (usize)(fsize * 1.5);
  std::vector<uptr> hashmap(mapsize);

  u64 current_index = 0, next_index = 0;

  auto t1 = high_resolution_clock::now();

  _mm_lfence();
  _mm_prefetch((char *)hashmap.data(), _MM_HINT_T1);

  HH_ALIGNAS(32) const highwayhash::HHKey key = {1, 2, 3, 4};
  while (current_index < fsize) [[likely]]
  {
    next_index = current_index + fast_getline((const char *)((uptr)ipt_buf.data() + current_index));

    highwayhash::HHStateT<HH_TARGET_AVX2> state(key);
    highwayhash::HighwayHashT(&state, ipt_buf.data() + current_index, next_index - current_index,
                              &result);

    auto hashvalue = result % mapsize;
    while (true) [[unlikely]]
    {
      if (hashmap[hashvalue] == 0) [[likely]]
      {
        hashmap[hashvalue] = (uptr)(ipt_buf.data() + current_index);
        ++counter;
        break;
      }
      if (strcmp_(ipt_buf.data() + current_index, (const char *)(void *)hashmap[hashvalue],
                  next_index - current_index) == true) [[unlikely]]
      {
        break;
      }
      hashvalue = (hashvalue + 1) % mapsize;
    }
    current_index = next_index + 1;
  }

  // auto after = rdtsc();
  auto t2 = high_resolution_clock::now();

  /* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;

  std::cout << "unique lines: " << counter << "  " << (ms_double.count() / 1000.f) << "s\n";
}