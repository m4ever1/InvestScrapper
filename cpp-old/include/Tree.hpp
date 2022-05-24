#pragma once
#include "PPCTreeNode.hpp"
#include "Item.hpp"
#include "NodeListTreeNode.hpp"

#include <vector>
#include <map>
#include <cstring>

class Tree {
private:
    int **bf;
    int bf_cursor;
    int bf_size;
    int bf_col;
    int bf_currentSize;

    NodeListTreeNode nlRoot;

    int numOfFItem;
    const std::vector<std::vector<Item>>& myTransactionsVector;
    std::shared_ptr<std::shared_ptr<PPCTreeNode>[]> myHeadTable;
    std::shared_ptr<int[]> myItemsetCount;
    std::shared_ptr<int[]> myHeadTableLen;
    const int& myMinSup;

    std::shared_ptr<NodeListTreeNode> isk_itemSetFreq(NodeListTreeNode& ni, NodeListTreeNode& nj, int level, std::shared_ptr<NodeListTreeNode> lastChild, int &sameCount);

    void traverse(std::shared_ptr<NodeListTreeNode> curNode, std::shared_ptr<NodeListTreeNode> curRoot, int level, int sameCount);
public:
    PPCTreeNode myRoot;
    void initializeTree();
    bool buildTree();
    void start_traversing();
    Tree(const std::vector<std::vector<Item>>&, const int&);
    ~Tree();
};