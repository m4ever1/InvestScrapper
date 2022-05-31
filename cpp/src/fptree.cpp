#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <limits>

#include "fptree.hpp"

const float IWISupport(std::set<Pattern>& itemset)
{
    float IWI = std::numeric_limits<float>::max();
    for(const Pattern& pat : itemset)
    {
        if (pat.second < IWI)
        {
            IWI = pat.second;
        }        
    }
    return IWI;
}

std::set<Pattern> getUnionWithHeaderEntry(const std::set<Pattern>& prefix, const std::pair<const Item, std::shared_ptr<FPNode>>& i)
{
    
    std::set<Pattern> result = prefix;
    std::set<Item> convertedToSet = {i.first};
    Pattern toInsert = std::make_pair(convertedToSet, i.second->frequency);
    result.insert(toInsert);
    return result;
}



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

FPTree::FPTree(const std::vector<Transaction>& transactions, const float& minimum_support_threshold_in) :
    root( std::make_shared<FPNode>( Item("ROOT", (float) 0, "ROOT"), nullptr, 0 ) ), header_table(),
    minimum_support_threshold( minimum_support_threshold_in )
{
    // scan the transactions counting the frequency of each item
    std::map<itemName, float> iwiSupportByItem;
    for ( const Transaction& transaction : transactions ) {
        for ( const Item& item : transaction ) {
            iwiSupportByItem[item.myName] += item.myExternalUtil;
        }
    }

    // keep only items which have a frequency greater or equal than the minimum support threshold
    // for ( auto it = iwiSupportByItem.cbegin(); it != iwiSupportByItem.cend(); ) 
    // {
    //     const float itemIWISupp = (*it).second;
    //     if ( itemIWISupp < minimum_support_threshold ) { iwiSupportByItem.erase( it++ ); }
    //     else { ++it; }
    // }

    // order items by decreasing frequency
    struct frequency_comparator
    {
        bool operator()(const std::pair<itemName, float> &lhs, const std::pair<itemName, float> &rhs) const
        {
            // 1st: compare the float values of each pair, return true if lhs is greater than rhs
            // if lhs is the same nr as rhs, compare the strings lexicographically, and return
            // true if lhs' string is lexicographcally greater than rhs'
            return std::tie(lhs.second, lhs.first) > std::tie(rhs.second, rhs.first);
        }
    };
    std::set<std::pair<itemName, float>, frequency_comparator> items_ordered_by_frequency(iwiSupportByItem.cbegin(), iwiSupportByItem.cend());

    // start tree construction

    // scan the transactions again
    for ( const Transaction& transaction : transactions ) 
    {
        std::list<EquivTrans> listOfEquivTrans = FPNode::convertToEquivTrans(transaction);

        // select and sort the frequent items in transaction according to the order of items_ordered_by_frequency
        for ( const auto& equivTransaction : listOfEquivTrans ) 
        {
            const float& weight = equivTransaction.myWeight;
            auto curr_fpnode = root;
            for(const Item& item : equivTransaction.myItems)
            {
                //WARNING: This version skips this step: ---
                    // -----> check if item is contained in the current transaction
                    // This is because our database has the same items in every transaction
                if ( true/*std::find( transaction.cbegin(), transaction.cend(), item ) != transaction.cend() */) 
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
                        curr_fpnode_child->frequency += weight;

                        // advance to the next node of the current transaction
                        curr_fpnode = curr_fpnode_child;
                    }
                }
            }
        }
    }
}

bool FPTree::empty() const
{
    assert( root );
    return root->children.size() == 0;
}


bool contains_single_path(const std::shared_ptr<FPNode>& fpnode)
{
    assert( fpnode );
    if ( fpnode->children.size() == 0 ) { return true; }
    if ( fpnode->children.size() > 1 ) { return false; }
    return contains_single_path( fpnode->children.front() );
}
bool contains_single_path(const FPTree& fptree)
{
    return fptree.empty() || contains_single_path( fptree.root );
}

std::set<Pattern> IWIMining(const FPTree& fptree, const float& minSup, const std::set<Pattern>& prefix)
{
    if ( fptree.empty() ) { return {}; }

    // if ( contains_single_path( fptree ) ) {
    //     // generate all possible combinations of the items in the tree

    //     std::set<Pattern> single_path_patterns;

    //     // for each node in the tree
    //     assert( fptree.root->children.size() == 1 );
    //     auto curr_fpnode = fptree.root->children.front();
    //     while ( curr_fpnode ) {
    //         const Item& curr_fpnode_item = curr_fpnode->item;
    //         const uint64_t curr_fpnode_frequency = curr_fpnode->frequency;

    //         // add a pattern formed only by the item of the current node
    //         Pattern new_pattern{ { curr_fpnode_item }, curr_fpnode_frequency };
    //         single_path_patterns.insert( new_pattern );

    //         // create a new pattern by adding the item of the current node to each pattern generated until now
    //         for ( const Pattern& pattern : single_path_patterns ) {
    //             Pattern new_pattern{ pattern };
    //             new_pattern.first.insert( curr_fpnode_item );
    //             assert( curr_fpnode_frequency <= pattern.second );
    //             new_pattern.second = curr_fpnode_frequency;

    //             single_path_patterns.insert( new_pattern );
    //         }

    //         // advance to the next node until the end of the tree
    //         assert( curr_fpnode->children.size() <= 1 );
    //         if ( curr_fpnode->children.size() == 1 ) { curr_fpnode = curr_fpnode->children.front(); }
    //         else { curr_fpnode = nullptr; }
    //     }

    //     return single_path_patterns;
    // }
    // else {
    //     // generate conditional fptrees for each different item in the fptree, then join the results

    std::set<Pattern> F;
    std::set<Pattern> I;

        // for each item in the fptree
    for ( const auto& pair : fptree.header_table ) 
    {
        const Item& curr_item = pair.first;

        std::set<Pattern> I = getUnionWithHeaderEntry(prefix, pair);

        if (IWISupport(I) <= minSup)
        {
            F.merge(I);
        }
        
    //     // build the conditional fptree relative to the current item

    //     // start by generating the conditional pattern base
    //     std::vector<TransformedPrefixPath> conditional_pattern_base;

    //     // for each path in the header_table (relative to the current item)
    //     auto path_starting_fpnode = pair.second;
    //     while ( path_starting_fpnode )
    //     {
    //         // construct the transformed prefix path

    //         // each item in the transformed prefix path has the same frequency (the frequency of path_starting_fpnode)
    //         const float path_starting_fpnode_frequency = path_starting_fpnode->frequency;

    //         auto curr_path_fpnode = path_starting_fpnode->parent.lock();
    //         // check if curr_path_fpnode is already the root of the fptree
    //         if ( curr_path_fpnode->parent.lock() ) {
    //             // the path has at least one node (excluding the starting node and the root)
    //             TransformedPrefixPath transformed_prefix_path{ {}, path_starting_fpnode_frequency };

    //             while ( curr_path_fpnode->parent.lock() ) 
    //             {
    //                 assert( curr_path_fpnode->frequency >= path_starting_fpnode_frequency );
    //                 transformed_prefix_path.first.push_back( curr_path_fpnode->item );

    //                 // advance to the next node in the path
    //                 curr_path_fpnode = curr_path_fpnode->parent.lock();
    //             }

    //             conditional_pattern_base.push_back( transformed_prefix_path );
    //         }

    //         // advance to the next path
    //         path_starting_fpnode = path_starting_fpnode->node_link;
    //     }

    //     // generate the transactions that represent the conditional pattern base
    //     std::vector<Transaction> conditional_fptree_transactions;
    //     for ( const TransformedPrefixPath& transformed_prefix_path : conditional_pattern_base ) 
    //     {
    //         const std::vector<Item>& transformed_prefix_path_items = transformed_prefix_path.first;
    //         const float transformed_prefix_path_items_frequency = transformed_prefix_path.second;

    //         Transaction transaction = transformed_prefix_path_items;

    //         // add the same transaction transformed_prefix_path_items_frequency times
    //         for ( auto i = 0; i < transformed_prefix_path_items_frequency; ++i ) {
    //             conditional_fptree_transactions.push_back( transaction );
    //         }
    //     }

    //     // build the conditional fptree relative to the current item with the transactions just generated
    //     const FPTree conditional_fptree( conditional_fptree_transactions, fptree.minimum_support_threshold );
    //     // call recursively fptree_growth on the conditional fptree (empty fptree: no patterns)
    //     std::set<Pattern> conditional_patterns = fptree_growth( conditional_fptree );

    //     // construct patterns relative to the current item using both the current item and the conditional patterns
    //     std::set<Pattern> curr_item_patterns;

    //     // the first pattern is made only by the current item
    //     // compute the frequency of this pattern by summing the frequency of the nodes which have the same item (follow the node links)
    //     float curr_item_frequency = 0;
    //     auto fpnode = pair.second;
    //     while ( fpnode ) {
    //         curr_item_frequency += fpnode->frequency;
    //         fpnode = fpnode->node_link;
    //     }
    //     // add the pattern as a result
    //     Pattern pattern{ { curr_item }, curr_item_frequency };
    //     curr_item_patterns.insert( pattern );

    //     // the next patterns are generated by adding the current item to each conditional pattern
    //     for ( const Pattern& pattern : conditional_patterns ) {
    //         Pattern new_pattern{ pattern };
    //         new_pattern.first.insert( curr_item );
    //         // assert( curr_item_frequency >= pattern.second );
    //         new_pattern.second = pattern.second;

    //         curr_item_patterns.insert( { new_pattern } );
    //     }

    //     // join the patterns generated by the current item with all the other items of the fptree
    //     multi_path_patterns.insert( curr_item_patterns.cbegin(), curr_item_patterns.cend() );
    // }

    // return multi_path_patterns;
    }
}
