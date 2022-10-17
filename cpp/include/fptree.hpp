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

class FPTree;

using TransformedPrefixPath = std::pair<std::vector<Item>, int>;
using Pattern = std::pair<std::set<Item>, int>;



const int IWISupport(std::set<Pattern>& itemset);
class FPNode {
public:
    const Item item;
    int frequency;
    std::shared_ptr<FPNode> node_link;
    std::weak_ptr<FPNode> parent;
    std::vector<std::shared_ptr<FPNode>> children;

    FPNode(const Item&, const std::shared_ptr<FPNode>&, const int);

    static std::list<EquivTrans> convertToEquivTrans(Transaction trans);
};

class FPTree {
public:

    std::shared_ptr<FPNode> root;
    std::map<Item, std::shared_ptr<FPNode>> header_table;
    int minimum_support_threshold;
    std::set<Item> itemsToKeep;
    FPTree(const std::vector<Transaction>&, const int&, std::map<Item, int>& iwiSupportByItem);
    FPTree(std::vector<EquivTrans>&, const int&);
    void printTree();
    void pruneItems();
    bool empty() const;
private:
    void DFS_prune(std::vector<Item>,const std::shared_ptr<FPNode>&);
    void BFS_print(const std::shared_ptr<FPNode>&, int level, std::map<int, std::string>& printOuts);
    void removeNode(const std::shared_ptr<FPNode>& nodeToRemove);
    void init();
};


#endif  // FPTREE_HPP

