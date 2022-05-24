#pragma once
#include <memory>

class NodeListTreeNode {
public:
	int label;
	std::shared_ptr<NodeListTreeNode> firstChild;
	std::shared_ptr<NodeListTreeNode> next;
	int support;
	int NLStartinBf;
	int NLLength;
	int NLCol;
    NodeListTreeNode();
    ~NodeListTreeNode();
};