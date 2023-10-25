#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include <windows.h>
#include <cmath>

#include "common.h"

namespace xenon_file
{
static usize
get_fsize(PCTSTR _path)
{
  HANDLE hFile{
      CreateFile(_path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, 0)};
  if (hFile == INVALID_HANDLE_VALUE)
  {
    printf("Error opening file: %s\n", _path);
    return FALSE;
  }
  return (usize)GetFileSize(hFile, 0);
}

static bool
read_file(PCTSTR _path, std::string& _buf) noexcept
{
  // nice way to avoid a goto and its practically as fast
  constexpr auto _out = [](bool _res, const HANDLE& _file, const std::vector<OVERLAPPED>& _ovls,
                           const u32& _div, const std::string& _buf)
  {
    for (usize i{0}; i != _div; ++i)
    {
      if (_ovls[i].hEvent)
      {
        CloseHandle(_ovls[i].hEvent);
      }
    }
    CloseHandle(_file);
    constexpr static usize cacheline_size{64};
    for (usize i{0}; i != _buf.size() / cacheline_size; ++i)
    {
      _mm_prefetch((b8*)&_buf[i * 64u], _MM_HINT_NTA);
    }
    return _res;
  };

  u32 _num_reads{0};
  HANDLE _file{
      CreateFile(_path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, 0)};

  // check if the handle is valid so if the opening of the
  // file was succesful
  if (_file == INVALID_HANDLE_VALUE)
  {
    return false;
  }
  // obtain file size in bytes
  u32 _fsize{GetFileSize(_file, 0u)};
  u32 _div{(u32)std::log2(_fsize) / 2u};

  if (_buf.size() <= _fsize) [[unlikely]]
  {
    _buf.resize(_fsize);
  }

  // get the first divisor that the size is
  // cleanly divisible through
  // @todo: prime numbers lol, also we dont
  //    always want the same amount of "threads"
  while ((_fsize / _div) * _div != _fsize)
  {
    _div++;
  }
  // @note:   chunk size for async reading
  //          pass this to ReadFile for every handle
  u32 _blk_size{_fsize / _div}; // size of each block, in bytes

  std::vector<OVERLAPPED> _ovls(_div);
  std::vector<HANDLE> _events(_div);

  for (usize i{0}; i != _div; ++i)
  {
    _ovls[i].Offset = (DWORD)(i * _blk_size);
    if (_ovls[i].hEvent = CreateEvent(0u, true, false, 0u); !_ovls[i].hEvent)
    {
      return _out(false, _file, _ovls, _div, _buf);
    }
    if (!ReadFile(_file, &_buf[i * _blk_size], _blk_size, 0, &_ovls[i]))
    {
      if (GetLastError() != ERROR_IO_PENDING)
      {
        CancelIo(_file);
        return _out(false, _file, _ovls, _div, _buf);
      }
    }
  }
  // make a buffer of events
  for (usize i{0}; i != _div; ++i)
  {
    _events[i] = _ovls[i].hEvent;
  }
  _num_reads = _div;
  // retrieve the events
  do
  {
    u32 _wait_res{WaitForMultipleObjects(_num_reads, &_events[0], false, INFINITE)};
    if (_wait_res == WAIT_FAILED)
    {
      CancelIo(_file);
      return _out(false, _file, _ovls, _div, _buf);
    }
    if ((_wait_res >= WAIT_OBJECT_0) && (_wait_res < (WAIT_OBJECT_0 + _num_reads)))
    {
      --_num_reads;
      if (_num_reads == 0) break;
    }
  } while (true);
  return _out(false, _file, _ovls, _div, _buf);
}
} // namespace xenon_file