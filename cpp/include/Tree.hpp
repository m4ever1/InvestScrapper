#pragma once
#include "PPCTreeNode.hpp"
#include "Item.hpp"
#include <vector>

class Tree {
private:
    const std::vector<std::vector<Item>>& myTransactionsVector;

public:
    PPCTreeNode myRoot;


    bool buildTree();
    Tree(const std::vector<std::vector<Item>>&);
    ~Tree();
};