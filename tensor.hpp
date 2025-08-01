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
template<typename T>
class Tensor {
    vector<T>      data_;
    vector<size_t> shape_;
    vector<size_t> strides_;

    inline static constexpr size_t total(const vector<size_t>& s)
    {
        return accumulate(s.begin(), s.end(), size_t{1}, multiplies<size_t>());
    }
    inline void bake_strides()
    {
        strides_.assign(shape_.size(), 1);
        for (size_t i = shape_.size(); i-- > 1;)
            strides_[i - 1] = strides_[i] * shape_[i];
    }
    inline size_t flat(const vector<size_t>& idx) const
    {
        const size_t*  i = idx.data();
        const size_t*  s = strides_.data();
        size_t off = 0, r = shape_.size();
        while (r--) { off += (*i++) * (*s++); }
        return off;
    }

public:
    Tensor() = default;
    explicit Tensor(const vector<size_t>& dims, const T& v = T())
        : data_(total(dims), v), shape_(dims) { bake_strides(); }

    inline T&       operator()(const vector<size_t>& idx)       { return data_[flat(idx)]; }
    inline const T& operator()(const vector<size_t>& idx) const { return data_[flat(idx)]; }

    const vector<size_t>& shape()   const { return shape_;   }
    const vector<size_t>& strides() const { return strides_; }
    size_t size() const { return data_.size(); }
    void fill(const T& v) { std::fill(data_.begin(), data_.end(), v); }

    /* needed by einsum */
    const vector<T>&       _data()   const { return data_; }
    vector<T>&             _data()         { return data_; }
    const vector<size_t>&  _strides() const { return strides_; }

    /* save / load */
    void save(const string& f) const
    {
        ofstream out(f, ios::binary);
        size_t r = shape_.size();
        out.write((char*)&r, sizeof(r));
        out.write((char*)shape_.data(), sizeof(size_t)*r);
        out.write((char*)data_.data(),  sizeof(T)*data_.size());
    }
    void load(const string& f)
    {
        ifstream in(f, ios::binary);
        size_t r; in.read((char*)&r, sizeof(r));
        shape_.resize(r);
        in.read(reinterpret_cast<char*>(shape_.data()), sizeof(size_t)*r);
        data_.assign(total(shape_), T());    
    }
};

#endif 