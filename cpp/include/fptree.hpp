#ifndef FPTREE_HPP
#define FPTREE_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <list>

#include "EquivTrans.hpp"

using TransformedPrefixPath = std::pair<std::vector<Item>, float>;
using Pattern = std::pair<std::set<Item>, float>;


class FPNode {
public:
    const Item item;
    float frequency;
    std::shared_ptr<FPNode> node_link;
    std::weak_ptr<FPNode> parent;
    std::vector<std::shared_ptr<FPNode>> children;

    FPNode(const Item&, const std::shared_ptr<FPNode>&, const float);

    static std::list<EquivTrans> convertToEquivTrans(Transaction trans);
};

class FPTree {
public:

    std::shared_ptr<FPNode> root;
    std::map<Item, std::shared_ptr<FPNode>> header_table;
    float minimum_support_threshold;

    FPTree(const std::vector<Transaction>&, const float&);

    bool empty() const;
};


#endif  // FPTREE_HPP

