#include "Tree.hpp"

Tree::Tree(const std::vector<std::vector<Item>>& vectorIn, const int& minSup) : 
myTransactionsVector(vectorIn), 
myMinSup(minSup)
{
    myRoot.label = -1;
}

Tree::~Tree() {

}
bool Tree::buildTree()
{
    int numberOfTransactions = myTransactionsVector.size();
    int numberOfUniqueItems = myTransactionsVector[0].size();
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
                if(child->label == transaction[curPos].myLabel)
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
            ppcNode->label = item.myLabel;
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
    //TODO: Consider only assets which have positive percentages at the last timestamp
    myHeadTable = std::shared_ptr<std::shared_ptr<PPCTreeNode>[]>(new std::shared_ptr<PPCTreeNode>[numberOfUniqueItems]);
    myHeadTableLen = std::shared_ptr<int[]>(new int[numberOfUniqueItems]);

	myItemsetCount = std::shared_ptr<int[]>(new int[(numberOfUniqueItems-1) * numberOfUniqueItems / 2]);
	memset(myItemsetCount.get(), 0, sizeof(int) * (numberOfUniqueItems-1) * numberOfUniqueItems / 2);

	std::memset(myHeadTableLen.get(), 0, sizeof(int) * numberOfUniqueItems);

	// TODO: CHANGE MAP INTO ARRAYS

	std::shared_ptr<PPCTreeNode> root = myRoot.firstChild;
	int pre = 0, last = 0;
	while(root != NULL)
	{
		root->foreIndex = pre;
		pre++;

		if(myHeadTable[root->label] == NULL)
		{	
            myHeadTable[root->label] = root;

			// tempHead[root->name] = root;
		}
		else
		{
			// tempHead[root->label]->labelSibling = root;
			// tempHead[root->label] = root;		
		}

        myHeadTableLen[root->label]++;
        
		std::shared_ptr<PPCTreeNode> temp = root->father;
		while(temp->label != -1)
		{
			myItemsetCount[root->label * (root->label - 1) / 2 + temp->label] += root->count;
			temp = temp->father;
		}
		if(root->firstChild != NULL)
			root = root->firstChild;
		else
		{
			//backvist
			root->backIndex=last;
			last++;
			if(root->rightSibling != NULL)
				root = root->rightSibling;
			else
			{
				root = root->father;
				while(root != NULL)
				{	
					//backvisit
					root->backIndex=last;
					last++;
					if(root->rightSibling != NULL)
					{
						root = root->rightSibling;
						break;
					}
					root = root->father;
				}
			}
		}
	}
}
