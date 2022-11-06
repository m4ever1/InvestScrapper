#pragma once
#include "Item.hpp"
#include <map>
#include <set>
#include "fptree.hpp"
#include <limits>
#include <cassert>
#include <unordered_set>
#include <iomanip>

class Utils {

public:
    Utils(const std::vector<Transaction>&, const int&, const int&);
    void countItemIWISupport(const std::vector<Transaction>&);
    ~Utils();
    std::set<Pattern> IWIMining(const FPTree& fptree, const int& minSup, int);
    std::map<Item, int>& getIwiSupportByItem();
    static void printEquivTrans(const std::list<EquivTrans>&);
    int calcDiversification(std::set<Item>);
    static void showProgress(const float& progress);
private:
    const std::vector<Transaction>& transactions;
    const int& minSup;
    std::map<Item, int> iwiSupportByItem;
    std::set<Pattern> getUnionWithHeaderEntry(const std::set<Pattern>& prefix, const std::pair<const Item, std::shared_ptr<FPNode>>& i);
    bool contains_single_path(const std::shared_ptr<FPNode>& fpnode);
    bool contains_single_path(const FPTree& fptree);
    int minDiv;
};