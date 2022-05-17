#pragma once
#include "PPCTreeNode.hpp"
#include "Item.hpp"
#include <vector>
#include <map>

class Tree {
private:
    const std::vector<std::vector<Item>>& myTransactionsVector;
    std::shared_ptr<int[]> myHeadTable;
    std::shared_ptr<int[]> myHeadTableLen;

public:
    PPCTreeNode myRoot;


    bool buildTree();
    Tree(const std::vector<std::vector<Item>>&);
    ~Tree();
};