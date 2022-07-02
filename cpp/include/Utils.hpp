#pragma once
#include "Item.hpp"
#include <map>
#include <set>
#include "fptree.hpp"
#include <limits>
#include <cassert>



class Utils {

public:
    Utils(const std::vector<Transaction>&, const float&);
    void countItemIWISupport(const std::vector<Transaction>&);
    ~Utils();
    std::set<Pattern> IWIMining(const FPTree& fptree, const float& minSup, const std::set<Pattern>& prefix);
    const std::map<Item, float>& getIwiSupportByItem();
    static void printEquivTrans(const std::list<EquivTrans>&);
private:
    const std::vector<Transaction>& transactions;
    const float& minSup;
    std::map<Item, float> iwiSupportByItem;
    const float IWISupport(std::set<Pattern>& itemset);
    std::set<Pattern> getUnionWithHeaderEntry(const std::set<Pattern>& prefix, const std::pair<const Item, std::shared_ptr<FPNode>>& i);
    bool contains_single_path(const std::shared_ptr<FPNode>& fpnode);
    bool contains_single_path(const FPTree& fptree);
};