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
    myHeadTable = std::make_shared<int[]>()
    // headTable = new PPCTreeNode*[numOfFItem];
	// memset(headTable, 0, sizeof(int*) * numOfFItem);
	// headTableLen = new int[numOfFItem];
	// memset(headTableLen, 0, sizeof(int) * numOfFItem);
	// PPCTreeNode **tempHead = new PPCTreeNode*[numOfFItem];
	
	// itemsetCount = new int[(numOfFItem-1) * numOfFItem / 2];
	// memset(itemsetCount, 0, sizeof(int) * (numOfFItem-1) * numOfFItem / 2);
	// TODO: CHANGE MAP INTO ARRAYS
    // TODO: FIGURE OUT WTF "numOfFItem" is...
	std::shared_ptr<PPCTreeNode> root = myRoot.firstChild;
	int pre = 0, last = 0;
	while(root != NULL)
	{
		root->foreIndex = pre;
		pre++;

		if(myHeadMap.find(root->name) == myHeadMap.end())
		{	
			myHeadMap.emplace(root->name, root);
			// tempHead[root->name] = root;
		}
		else
		{
			// tempHead[root->label]->labelSibling = root;
			// tempHead[root->label] = root;		
		}

        if (myHeadMapLen.find(root->name) == myHeadMapLen.end())
        {
            myHeadMapLen.emplace(root->name, 1);
        }
        else
        {
		    myHeadMapLen[root->name]++;
        }
        
    
		PPCTreeNode *temp = root->father;
		while(temp->label != -1)
		{
			itemsetCount[root->label * (root->label - 1) / 2 + temp->label] += root->count;
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
