#include "Tree.hpp"

Tree::Tree(const std::vector<std::vector<Item>>& vectorIn, const int& minSup) : 
myTransactionsVector(vectorIn), 
myMinSup(minSup)
{
    myRoot.label = -1;

    bf_size = 1000000;
	bf = new int*[100000];
	bf_currentSize = bf_size * 10;
	bf[0] = new int[bf_currentSize];

    bf_cursor = 0;
	bf_col = 0;

    nlRoot.label = myTransactionsVector.size();
	nlRoot.firstChild = NULL;
	nlRoot.next = NULL;
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


	std::shared_ptr<PPCTreeNode> root = myRoot.firstChild;  
	int pre = 0, last = 0;
	while(root != NULL)
	{
		root->foreIndex = pre;
		pre++;

		if(myHeadTable[root->label] == NULL)
		{	
            myHeadTable[root->label] = root;
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

void Tree::initializeTree()
{
	std::shared_ptr<NodeListTreeNode> lastChild = NULL;
	for(int t = myTransactionsVector.size() - 1; t >= 0; t--)
	{
		if(bf_cursor > bf_currentSize - myHeadTableLen[t] * 3)
		{
			bf_col++;
			bf_cursor = 0;
			bf_currentSize = 10 * bf_size;
			bf[bf_col] = new int[bf_currentSize];
		}

		std::shared_ptr<NodeListTreeNode> nlNode = std::make_shared<NodeListTreeNode>;
		nlNode->label = t;
		nlNode->support = 0;
		nlNode->NLStartinBf = bf_cursor;
		nlNode->NLLength = 0;
		nlNode->NLCol = bf_col;
		nlNode->firstChild = NULL;
		nlNode->next = NULL;
		std::shared_ptr<PPCTreeNode> ni = myHeadTable[t];
		while(ni != NULL)
		{
			nlNode->support+= ni->count;
			bf[bf_col][bf_cursor++] =  ni->foreIndex;
			bf[bf_col][bf_cursor++] =  ni->backIndex;
			bf[bf_col][bf_cursor++] =  ni->count;
			nlNode->NLLength++;
			ni = ni->labelSibling;
		}
		if(nlRoot.firstChild == NULL)
		{
			nlRoot.firstChild = nlNode;
			lastChild = nlNode;
		}
		else
		{
			lastChild->next = nlNode;
			lastChild = nlNode;
		}
	}
}

void Tree::start_traversing()
{
	int from_cursor = bf_cursor;
	int from_col = bf_col;
	int from_size = bf_currentSize;

    std::shared_ptr<NodeListTreeNode> curNode = nlRoot.firstChild;
	std::shared_ptr<NodeListTreeNode> next = NULL;

    while(curNode != NULL)
    {
        next = curNode->next;
        traverse(curNode, std::make_shared<NodeListTreeNode>(nlRoot), 1, 0);
    }
}

std::shared_ptr<NodeListTreeNode> Tree::isk_itemSetFreq(NodeListTreeNode& ni, NodeListTreeNode& nj, int level, std::shared_ptr<NodeListTreeNode> lastChild, int &sameCount)
{
	if(bf_cursor + ni.NLLength * 3 > bf_currentSize)
	{
		bf_col++;
		bf_cursor = 0;
		bf_currentSize = bf_size > ni.NLLength * 1000 ? bf_size : ni.NLLength * 1000;
		bf[bf_col] = new int[bf_currentSize];
	}
		
	std::shared_ptr<NodeListTreeNode> nlNode = std::make_shared<NodeListTreeNode>();
	nlNode->support = 0;
	nlNode->NLStartinBf = bf_cursor;
	nlNode->NLCol = bf_col;
	nlNode->NLLength = 0;
	
	int cursor_i = ni.NLStartinBf;
	int cursor_j = nj.NLStartinBf;
	int col_i = ni.NLCol;
	int col_j = nj.NLCol;
	int last_cur = -1;
	while(cursor_i < ni.NLStartinBf + ni.NLLength * 3 && cursor_j < nj.NLStartinBf + nj.NLLength * 3)
	{
		if(bf[col_i][cursor_i] > bf[col_j][cursor_j] && bf[col_i][cursor_i + 1] < bf[col_j][cursor_j + 1])
		{
			if(last_cur == cursor_j)
			{
				bf[bf_col][bf_cursor - 1] += bf[col_i][cursor_i + 2];
			}
			else
			{
				bf[bf_col][bf_cursor++] =  bf[col_j][cursor_j];
				bf[bf_col][bf_cursor++] =  bf[col_j][cursor_j + 1];
				bf[bf_col][bf_cursor++] =  bf[col_i][cursor_i + 2];
				nlNode->NLLength++;
			}
			nlNode->support += bf[col_i][cursor_i + 2];
			last_cur = cursor_j;
			cursor_i += 3;
		}
		else if(bf[col_i][cursor_i] < bf[col_j][cursor_j])
		{
			cursor_i += 3;
		}
		else if(bf[col_i][cursor_i + 1] > bf[col_j][cursor_j + 1])
		{
			cursor_j += 3;
		}
	}
	if(nlNode->support >= myMinSup)
	{
		if(ni.support == nlNode->support)// && nlNode->NLLength == 1)
		{
			bf_cursor = nlNode->NLStartinBf;
		}
		else
		{
			nlNode->label = nj.label;
			nlNode->firstChild = NULL;
			nlNode->next = NULL;
			if(ni.firstChild == NULL)
			{
				ni.firstChild = nlNode;
				lastChild = nlNode;
			}
			else
			{
				lastChild->next = nlNode;
				lastChild = nlNode;
			}
		}
		return lastChild;
	}
	else
	{
		bf_cursor = nlNode->NLStartinBf;
	}
	return lastChild;
}

void Tree::traverse(std::shared_ptr<NodeListTreeNode> curNode, std::shared_ptr<NodeListTreeNode> curRoot, int level, int sameCount)
{
	std::shared_ptr<NodeListTreeNode> sibling = curNode->next;
	std::shared_ptr<NodeListTreeNode> lastChild = NULL;
	while(sibling != NULL)
	{	
		if(level >1 || (level == 1 && myItemsetCount[(curNode->label-1) * curNode->label/2 + sibling->label] >= myMinSup))
		lastChild = isk_itemSetFreq(curNode, sibling, level, lastChild, sameCount);
		sibling = sibling->next;
	}
	
	resultCount += pow(2.0, sameCount);
	nlLenSum += pow(2.0, sameCount) * curNode->NLLength;

	if(dump == 1)
	{
		result[resultLen++] = curNode->label;
		for(int i = 0; i < resultLen; i++)
			fprintf(out, "%d ", item[result[i]].index);
		fprintf(out, "(%d %d)", curNode->support, curNode->NLLength);
		for(int i = 0; i < sameCount; i++)
			fprintf(out, " %d", item[sameItems[i]].index);
		fprintf(out, "\n");
	}
	nlNodeCount++;
	
	int from_cursor = bf_cursor;
	int from_col = bf_col;
	int from_size = bf_currentSize;
	NodeListTreeNode *child = curNode->firstChild;
	NodeListTreeNode *next = NULL;
	while(child != NULL)
	{
		next = child->next;
		traverse(child, curNode, level+1, sameCount);
		for(int c = bf_col; c > from_col; c--)
			delete[] bf[c];
		bf_col = from_col;
		bf_cursor = from_cursor;
		bf_currentSize = from_size;
		child = next;
	}
}