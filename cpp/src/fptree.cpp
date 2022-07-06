#include <algorithm>
#include <cstdint>
#include <cassert>
#include <utility>
#include "fptree.hpp"
#include "Utils.hpp"

FPNode::FPNode(const Item& item, const std::shared_ptr<FPNode>& parent, const float weight) :
    item( item ), frequency( weight ), node_link( nullptr ), parent( parent ), children()
{
}

std::list<EquivTrans> FPNode::convertToEquivTrans(Transaction S)
{
    std::sort(S.begin(), S.end(), [] (const Item& lhs, const Item& rhs){return lhs.myExternalUtil < rhs.myExternalUtil;});
    Transaction copyOfS = S;
    int p = std::unique(copyOfS.begin(), copyOfS.end(), 
            [] (const Item& lhs, const Item& rhs){return lhs.myExternalUtil == rhs.myExternalUtil;}) - copyOfS.begin();
    float wref = 0;
    std::list<EquivTrans> equivTransVec;

    while(equivTransVec.size() != p)
    {
        float extUtil = S.front().myExternalUtil;
        float wout = extUtil - wref;
        wref = extUtil;
        
        equivTransVec.push_back(EquivTrans(S, wout));

        for(Transaction::const_iterator it = S.begin(); it < S.end(); it++)
        {
            if((*it).myExternalUtil > wref)
            {
                S.erase(S.begin(), it);
                break;
            }
        } 
    }
    return equivTransVec;
}

FPTree::FPTree(std::vector<EquivTrans>& transactions, const float& minimum_support_threshold) :
    root( std::make_shared<FPNode>( Item("ROOT", (float) 0, "ROOT"), nullptr, 0 ) ), header_table(),
    minimum_support_threshold( minimum_support_threshold )
{
    // scan the transactions counting the frequency of each item

    // order items by decreasing frequency

    std::sort(transactions.begin(), transactions.end(), 
    [] (const EquivTrans& lhs, const EquivTrans& rhs){return lhs.myWeight > rhs.myWeight;});

    // keep only items which have a frequency greater or equal than the minimum support threshold
    // for ( auto it = iwiSupportByItem.cbegin(); it != iwiSupportByItem.cend(); ) 
    // {
    //     const float itemIWISupp = (*it).second;
    //     if ( itemIWISupp < minimum_support_threshold ) { iwiSupportByItem.erase( it++ ); }
    //     else { ++it; }
    // }


    // start tree construction

    // scan the transactions again
    for ( const EquivTrans& transaction : transactions ) 
    {
        auto curr_fpnode = root;
        // select and sort the frequent items in transaction according to the order of items_ordered_by_frequency
        const float& weight = transaction.myWeight;

        for(const auto& pair : transaction.myItems)
        {
            const Item& item = pair;
            // check if item is contained in the current transaction
            if (true) 
            {
                // insert item in the tree

                // check if curr_fpnode has a child curr_fpnode_child such that curr_fpnode_child.item = item
                const auto it = std::find_if(
                    curr_fpnode->children.cbegin(), curr_fpnode->children.cend(),  [item](const std::shared_ptr<FPNode>& fpnode) {
                        return fpnode->item.myName == item.myName;
                } );
                if ( it == curr_fpnode->children.cend() ) 
                {
                    // the child doesn't exist, create a new node
                    const auto curr_fpnode_new_child = std::make_shared<FPNode>( item, curr_fpnode, weight );

                    // add the new node to the tree
                    curr_fpnode->children.push_back( curr_fpnode_new_child );

                    // update the node-link structure
                    if ( header_table.count( curr_fpnode_new_child->item ) ) 
                    {
                        auto prev_fpnode = header_table[curr_fpnode_new_child->item];
                        while ( prev_fpnode->node_link ) 
                        { 
                            prev_fpnode = prev_fpnode->node_link; 
                        }
                        prev_fpnode->node_link = curr_fpnode_new_child;
                    }
                    else 
                    {
                        header_table[curr_fpnode_new_child->item] = curr_fpnode_new_child;
                    }

                    // advance to the next node of the current transaction
                    curr_fpnode = curr_fpnode_new_child;
                }
                else 
                {
                    // the child exist, increment its frequency
                    auto curr_fpnode_child = *it;
                    curr_fpnode_child->frequency+= weight;

                    // advance to the next node of the current transaction
                    curr_fpnode = curr_fpnode_child;
                }
            }
        }
    }
}

FPTree::FPTree(const std::vector<Transaction>& transactions, const float& minimum_support_threshold_in, const std::map<Item, float>& iwiSupportByItem) :
    root( std::make_shared<FPNode>( Item("ROOT", (float) 0, "ROOT"), nullptr, 0 ) ), header_table(),
    minimum_support_threshold( minimum_support_threshold_in )
{

    // keep only items which have a frequency greater or equal than the minimum support threshold
    // for ( auto it = iwiSupportByItem.cbegin(); it != iwiSupportByItem.cend(); ) 
    // {
    //     const float itemIWISupp = (*it).second;
    //     if ( itemIWISupp < minimum_support_threshold ) { iwiSupportByItem.erase( it++ ); }
    //     else { ++it; }
    // }

    // order items by decreasing frequency
    struct iwiSupportComparator
    {
        bool operator()(const std::pair<Item, float> &lhs, const std::pair<Item, float> &rhs) const
        {
            // 1st: compare the float values of each pair, return true if lhs is greater than rhs
            // if lhs is the same nr as rhs, compare the strings lexicographically, and return
            // true if lhs' string is lexicographcally greater than rhs'
            return std::tie(lhs.second, lhs.first.myName) > std::tie(rhs.second, rhs.first.myName);
        }
    };
    std::set<std::pair<Item, float>, iwiSupportComparator> itemsOrderedByIwi(iwiSupportByItem.cbegin(), iwiSupportByItem.cend());

    // start tree construction

    // scan the transactions again
    for ( const Transaction& transaction : transactions ) 
    {
        std::list<EquivTrans> listOfEquivTrans = FPNode::convertToEquivTrans(transaction);
        Utils::printEquivTrans(listOfEquivTrans);
        // select and sort the frequent items in transaction according to the order of items_ordered_by_frequency
        for ( const auto& equivTransaction : listOfEquivTrans ) 
        {
            const float& weight = equivTransaction.myWeight;
            auto curr_fpnode = root;
            for(const auto& pair : itemsOrderedByIwi)
            {

                // check if item is contained in the current transaction
                auto found_item = std::find( equivTransaction.myItems.begin(), equivTransaction.myItems.end(), pair.first );
                if ( found_item != equivTransaction.myItems.end() ) 
                {
                    const Item& item = *found_item;
                    // insert item in the tree

                    // check if curr_fpnode has a child curr_fpnode_child such that curr_fpnode_child.item = item
                    const auto it = std::find_if(
                        curr_fpnode->children.cbegin(), curr_fpnode->children.cend(),  [item](const std::shared_ptr<FPNode>& fpnode) {
                            return fpnode->item.myName == item.myName;
                    } );
                    if ( it == curr_fpnode->children.cend() ) 
                    {
                        // the child doesn't exist, create a new node
                        const auto curr_fpnode_new_child = std::make_shared<FPNode>( item, curr_fpnode, weight );

                        // add the new node to the tree
                        curr_fpnode->children.push_back( curr_fpnode_new_child );

                        // update the node-link structure
                        if ( header_table.count( curr_fpnode_new_child->item ) ) 
                        {
                            auto prev_fpnode = header_table[curr_fpnode_new_child->item];
                            while ( prev_fpnode->node_link ) 
                            { 
                                prev_fpnode = prev_fpnode->node_link; 
                            }
                            prev_fpnode->node_link = curr_fpnode_new_child;
                        }
                        else 
                        {
                            header_table[curr_fpnode_new_child->item] = curr_fpnode_new_child;
                        }

                        // advance to the next node of the current transaction
                        curr_fpnode = curr_fpnode_new_child;
                    }
                    else 
                    {
                        // the child exist, increment its frequency
                        auto curr_fpnode_child = *it;
                        curr_fpnode_child->frequency += weight;

                        // advance to the next node of the current transaction
                        curr_fpnode = curr_fpnode_child;
                    }
                }
            }
        }
    }
}

void FPTree::printTree()
{
    std::map<int, std::string> printOuts;
    std::cout << "\t\troot" << std::endl;
    BFS_print(root, 0, printOuts);
    for(const auto& line : printOuts)
    {
        std::cout << line.second << std::endl;
    }
}

void FPTree::BFS_print(const std::shared_ptr<FPNode>& curr, int level, std::map<int, std::string>& printOuts)
{
    for(const auto& child : curr->children)
    {
        printOuts[level] += child->item.myName + ':' + std::to_string(child->frequency) + '\t';
        BFS_print(child, level++, printOuts);
    }
}

void FPTree::removeNode(const std::shared_ptr<FPNode>& nodeToRemove)
{
    const std::shared_ptr<FPNode> parent = nodeToRemove->parent.lock();
    for(const auto& child : nodeToRemove->children)
    {
        parent->children.push_back(child);
        child->parent = parent;
    }
    if (nodeToRemove->node_link)
    {
        removeNode(nodeToRemove->node_link);
    }
    
}

void FPTree::pruneItems()
{
    std::vector<Item> initPath;
    DFS_prune(initPath, root);
    for (auto it = header_table.cbegin(); it != header_table.cend();)
    {
        if(itemsToKeep.find(it->first) == itemsToKeep.end())
        {
            removeNode(it->second);
            it = header_table.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool FPTree::empty() const
{
    assert( root );
    return root->children.size() == 0;
}

void FPTree::DFS_prune(std::vector<Item> path, const std::shared_ptr<FPNode>& curr)
{
    path.push_back(curr->item);

    if (curr->children.empty())
    {
        if (curr->frequency > minimum_support_threshold)
        {
            itemsToKeep.insert(path.begin(), path.end());
        }
        return;
    }
    
    for (const std::shared_ptr<FPNode>& child : curr->children)
    {
        DFS_prune(path, child);
    }
    return;
}




