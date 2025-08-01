#ifndef EINSUM_HPP
#define EINSUM_HPP

#include "tensor.hpp"
#include <unordered_map>
#include <array>
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <algorithm>

/* ─────────────────────────────────────────
   Einstein-summation
   ───────────────────────────────────────── */
template<typename T, std::size_t Rank>
Tensor<T, Rank>
einsum(const std::string& eq, const std::vector< Tensor<T, Rank> >& ops)
{
    /* 1. split “lhs -> rhs” */
    const std::size_t arrow = eq.find("->");
    const std::string lhs   = eq.substr(0, arrow);
    std::string       rhs   = (arrow == std::string::npos)
                                ? "" : eq.substr(arrow + 2);

    /* 2. separate operand subscripts */
    std::vector<std::string> subs;
    std::size_t beg = 0;
    for (std::size_t i = 0; i <= lhs.size(); ++i)
        if (i == lhs.size() || lhs[i] == ',')
        { subs.push_back(lhs.substr(beg, i - beg)); beg = i + 1; }

    if (subs.size() != ops.size())
        throw std::invalid_argument("operand count mismatch with equation");

    /* 3. collect extents & label frequency */
    std::unordered_map<char, std::size_t> extent;
    std::unordered_map<char, int>         freq;
    for (std::size_t t = 0; t < ops.size(); ++t)
        for (std::size_t a = 0; a < subs[t].size(); ++a)
        {
            const char lbl = subs[t][a];
            const std::size_t dim = ops[t].shape()[a];
            if (extent.count(lbl) && extent[lbl] != dim)
                throw std::invalid_argument("dimension mismatch on label");
            extent[lbl] = dim; ++freq[lbl];
        }

    /* 4. implicit RHS (unique labels) */
    if (rhs.empty())
        for (const auto& kv : freq)
            if (kv.second == 1) rhs.push_back(kv.first);

    if (rhs.size() != Rank)
        throw std::invalid_argument("output rank must equal template Rank");

    /* 5. build output tensor */
    std::array<std::size_t, Rank> out_shape{};
    for (std::size_t i = 0; i < Rank; ++i) out_shape[i] = extent[rhs[i]];

    Tensor<T, Rank> out(out_shape);
    out.fill(T(0));

    /* 6. linear coords over ALL labels (sorted map) */
    std::vector<char>   labels;
    std::vector<std::size_t> bounds;
    for (const auto& kv : extent) { labels.push_back(kv.first);
                                    bounds.push_back(kv.second); }

    std::vector<std::size_t> coord(labels.size(), 0);

    /* quick access to strides */
    std::vector< const std::array<std::size_t, Rank>* > opstr;
    for (const auto& t : ops) opstr.push_back(&t._strides());
    const auto& ostr = out._strides();

    /* 7. recursion across label space */
    std::function<void(std::size_t)> rec = [&](std::size_t depth)
    {
        if (depth == labels.size())
        {
            /* gather & multiply operands */
            T prod(1);
            for (std::size_t t = 0; t < ops.size(); ++t)
            {
                const std::string& sub = subs[t];
                std::size_t flat = 0;
                for (std::size_t a = 0; a < sub.size(); ++a)
                {
                    const char lbl = sub[a];
                    const std::size_t idx =
                        coord[ std::find(labels.begin(), labels.end(), lbl)
                               - labels.begin() ];
                    flat += idx * (*opstr[t])[a];
                }
                prod *= ops[t]._data()[flat];
            }
            /* scatter to output */
            std::size_t fo = 0;
            for (std::size_t a = 0; a < Rank; ++a)
            {
                const char lbl = rhs[a];
                const std::size_t idx =
                    coord[ std::find(labels.begin(), labels.end(), lbl)
                           - labels.begin() ];
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
