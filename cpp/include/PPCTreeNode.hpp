#pragma once
#include <memory>

class PPCTreeNode {

public:
	std::string name;
	int label;
	std::shared_ptr<PPCTreeNode> firstChild;
	std::shared_ptr<PPCTreeNode> rightSibling;
	std::shared_ptr<PPCTreeNode> labelSibling;
	std::shared_ptr<PPCTreeNode> father;
	int count;
	int foreIndex;
	int backIndex;
    PPCTreeNode();
    ~PPCTreeNode();
};