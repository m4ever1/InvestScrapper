#include "Tree.hpp"

Tree::Tree(const std::vector<std::vector<Item>>& vectorIn) : myTransactionsVector(vectorIn)
{
    myRoot.name = -1;
}

Tree::~Tree() {

}
bool Tree::buildTree()
{
    for(const auto& transaction: myTransactionsVector)
    {
        int curPos = 0;
        // we are at the root node
        std::shared_ptr<PPCTreeNode> curRoot = std::make_shared<PPCTreeNode>(myRoot);
        std::shared_ptr<PPCTreeNode> rightSibling = NULL;
        while(curPos != transaction.size())
        {
            std::shared_ptr<PPCTreeNode> child = curRoot->firstChild;
            while(child)
            {				
                if(child->name == transaction[curPos].myName)
                {					
                    curPos++;
                    child->count++;
                    curRoot = child;
                    break;
                }
                if(child -> rightSibling == NULL)
                {
                    rightSibling = child;
                    child = NULL;
                    break;
                }
                child = child -> rightSibling;
            }			
            if(!child) 
            {
                break;
            }
        }
        for(const auto& item : transaction)
        {
            std::shared_ptr<PPCTreeNode> ppcNode = std::make_shared<PPCTreeNode>();
            ppcNode->name = item.myName;
            if(rightSibling != NULL)
            {
                rightSibling->rightSibling = ppcNode;
                rightSibling = NULL;
            }
            else
            {
                curRoot->firstChild = ppcNode;	
            }
            ppcNode->rightSibling = NULL;
            ppcNode->firstChild = NULL;
            ppcNode->father = curRoot;
            ppcNode->labelSibling = NULL;
            ppcNode->count = 1;
            curRoot = ppcNode;
        }
    }
}
