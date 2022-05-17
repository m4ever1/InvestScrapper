#pragma once
#include "PPCTreeNode.hpp"
#include "Item.hpp"
#include <vector>
#include <map>
#include <cstring>

class Tree {
private:
    int numOfFItem;
    const std::vector<std::vector<Item>>& myTransactionsVector;
    std::shared_ptr<std::shared_ptr<PPCTreeNode>[]> myHeadTable;
    std::shared_ptr<int[]> myItemsetCount;
    std::shared_ptr<int[]> myHeadTableLen;
    const int& myMinSup;

public:
    PPCTreeNode myRoot;


    bool buildTree();
    Tree(const std::vector<std::vector<Item>>&, const int&);
    ~Tree();
};