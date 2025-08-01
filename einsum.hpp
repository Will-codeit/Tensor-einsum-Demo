#ifndef EINSUM_HPP
#define EINSUM_HPP

#include "tensor.hpp"
#include <unordered_map>
#include <functional>

/* Einstein summation for Tensor<T>  */
template<typename T>
Tensor<T> einsum(const string& eq, const vector< Tensor<T> >& ops)
{
    /* Parse equation */
    size_t arrow = eq.find("->");
    const string lhs = eq.substr(0, arrow);
    string rhs = (arrow == string::npos) ? "" : eq.substr(arrow + 2);

    /* split LHS subscripts */
    vector<string> subs;
    size_t beg = 0;
    for (size_t i = 0; i <= lhs.size(); ++i)
        if (i == lhs.size() || lhs[i] == ',') { subs.push_back(lhs.substr(beg, i - beg)); beg = i + 1; }

    if (subs.size() != ops.size())
        throw invalid_argument("operand count mismatch with equation");

    /* label bookkeeping */
    unordered_map<char,size_t> extent;
    unordered_map<char,int>    freq;
    for (size_t t = 0; t < ops.size(); ++t)
        for (size_t a = 0; a < subs[t].size(); ++a)
        {
            char lbl = subs[t][a];
            size_t dim = ops[t].shape()[a];
            if (extent.count(lbl) && extent[lbl] != dim)
                throw invalid_argument("dimension mismatch on label");
            extent[lbl] = dim; ++freq[lbl];
        }

    /* implicit rhs */
    if (rhs.empty())
        for (auto& kv : freq) if (kv.second == 1) rhs.push_back(kv.first);

    /* build output tensor */
    vector<size_t> out_shape; out_shape.reserve(rhs.size());
    for (char c : rhs) out_shape.push_back(extent[c]);
    Tensor<T> out(out_shape); out.fill(T(0));

    /* 4. recurse over all labels */
    vector<char>   labels;
    vector<size_t> bounds;
    for (auto& kv : extent) { labels.push_back(kv.first); bounds.push_back(kv.second); }

    vector<size_t> coord(labels.size(),0);

    /* pre-grab strides */
    vector<const vector<size_t>*> opstr;
    for (auto& t : ops) opstr.push_back(&t._strides());
    const vector<size_t>& ostr = out._strides();

    function<void(size_t)> rec = [&](size_t depth)
    {
        if (depth == labels.size())
        {
            T prod = T(1);
            /* gather operands */
            for (size_t t = 0; t < ops.size(); ++t)
            {
                const string& sub = subs[t];
                size_t flat = 0;
                for (size_t a = 0; a < sub.size(); ++a)
                {
                    size_t idx = coord[find(labels.begin(), labels.end(), sub[a]) - labels.begin()];
                    flat += idx * (*opstr[t])[a];
                }
                prod *= ops[t]._data()[flat];
            }
            /* map to output */
            size_t fo = 0;
            for (size_t a = 0; a < rhs.size(); ++a)
            {
                size_t idx = coord[find(labels.begin(), labels.end(), rhs[a]) - labels.begin()];
                fo += idx * ostr[a];
            }
            out._data()[fo] += prod;
            return;
        }
        for (coord[depth]=0; coord[depth]<bounds[depth]; ++coord[depth])
            rec(depth+1);
    };
    rec(0);
    return out;
}

#endif