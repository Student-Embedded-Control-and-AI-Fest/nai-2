#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "nai_flash_port.h"
#include "nai_slotfs.h"

class NoodleBLFlashFile {
public:
    NoodleBLFlashFile() = default;

    static NoodleBLFlashFile openRead(const char *virtual_name)
    {
        NoodleBLFlashFile f;

        char slot = 0;
        char canonical[NAI_SLOT_NAME_BYTES]{};
        if (!nai_slotfs_virtual_name(virtual_name, &slot, canonical))
            return f;

        uint32_t addr = 0, size = 0;
        const bool staged = nai_slotfs_staged_read_allowed(slot);
        if (!nai_slotfs_lookup(slot, canonical, &addr, &size, staged))
            return f;

        f._open = true;
        f._base = addr;
        f._size = size;
        f._pos = 0;
        return f;
    }

    operator bool() const { return _open; }

    bool available() const
    {
        return _open && _pos < _size;
    }

    int read()
    {
        uint8_t b = 0;
        if (read(&b, 1) != 1) return -1;
        return b;
    }

    int peek()
    {
        if (!_open || _pos >= _size) return -1;
        uint8_t b = 0;
        if (!nai_flash_read(_base + _pos, &b, 1)) return -1;
        return b;
    }

    size_t read(uint8_t *dst, size_t n)
    {
        if (!_open || !dst || n == 0 || _pos >= _size) return 0;
        size_t remain = _size - _pos;
        if (n > remain) n = remain;
        if (!nai_flash_read(_base + _pos, dst, n)) return 0;
        _pos += (uint32_t)n;
        return n;
    }

    size_t readBytes(char *dst, size_t n)
    {
        return read((uint8_t *)dst, n);
    }

    size_t write(uint8_t) { return 0; }
    size_t write(const uint8_t *, size_t) { return 0; }
    void flush() {}

    void close()
    {
        _open = false;
        _base = _size = _pos = 0;
    }

    size_t size() const { return _open ? _size : 0; }
    size_t position() const { return _open ? _pos : 0; }

    bool seek(uint32_t pos)
    {
        if (!_open || pos > _size) return false;
        _pos = pos;
        return true;
    }

    size_t println(uint8_t) { return 0; }
    size_t println(float) { return 0; }
    size_t println(float, int) { return 0; }

private:
    bool _open = false;
    uint32_t _base = 0;
    uint32_t _size = 0;
    uint32_t _pos = 0;
};
