#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <vector>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <fstream>

using namespace std;

/* row-major */
template<typename T, size_t Rank>
class Tensor {
    
    vector<T>      data_;
    array<size_t, Rank> shape_;
    array<size_t, Rank> strides_;

    
    static constexpr size_t total(const array<size_t, Rank>& s)
    {
        size_t n = 1;
        for (auto v : s) n *= v;
        return n;
    }
    constexpr void bake_strides()
    {
        strides_[Rank - 1] = 1;
        for (size_t i = Rank - 1; i-- > 0;)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
    constexpr size_t flat(const array<size_t, Rank>& idx) const
    {
        size_t off = 0;
        for (size_t k = 0; k < Rank; ++k) off += idx[k] * strides_[k];
        return off;
    }

public:
    using value_type = T;
    constexpr Tensor() = default;
    explicit Tensor(const array<size_t, Rank>& dims, const T& v = T())
        : data_(total(dims), v), shape_(dims) { bake_strides(); }

    inline T&       operator()(const array<size_t, Rank>& idx)       { return data_[flat(idx)]; }
    inline const T& operator()(const array<size_t, Rank>& idx) const { return data_[flat(idx)]; }

    constexpr const auto& shape()   const noexcept { return shape_;   }
    constexpr const auto& strides() const noexcept { return strides_; }
    constexpr size_t size()         const noexcept { return data_.size(); }
    void fill(const T& v) { std::fill(data_.begin(), data_.end(), v); }

    /* needed by einsum */
    auto*       _data() noexcept       { return data_.data(); }
    const auto* _data() const noexcept { return data_.data(); }
    const auto& _strides() const       { return strides_; }

     /* ── save / load (binary) ─────────────────── */
    void save(const std::string& f) const
    {
        ofstream out(f, ios::binary);
        out.write(reinterpret_cast<const char*>(shape_.data()), sizeof(shape_));
        out.write(reinterpret_cast<const char*>(data_.data()),  sizeof(T)*data_.size());
    }
    void load(const std::string& f)
    {
        ifstream in(f, ios::binary);
        in.read(reinterpret_cast<char*>(shape_.data()), sizeof(shape_));
        bake_strides();
        data_.resize(total(shape_));
        in.read(reinterpret_cast<char*>(data_.data()), sizeof(T)*data_.size());
    }
};

#endif 
