#include "Utils.hpp"

bool Utils::contains_single_path(const std::shared_ptr<FPNode>& fpnode)
{
    assert( fpnode );
    if ( fpnode->children.size() == 0 ) { return true; }
    if ( fpnode->children.size() > 1 ) { return false; }
    return contains_single_path( fpnode->children.front() );
}
bool Utils::contains_single_path(const FPTree& fptree)
{
    return fptree.empty() || contains_single_path( fptree.root );
}

Utils::Utils(const std::vector<Transaction>& transactions_in, const int& minSup_in, const int& minDiv_in) : 
transactions(transactions_in), minSup(minSup_in), minDiv(minDiv_in)
{
    Utils::countItemIWISupport(transactions_in);
}

void Utils::countItemIWISupport(const std::vector<Transaction>& transactions)
{
    for ( const Transaction& transaction : transactions ) 
    {
        for ( const Item& item : transaction ) 
        {
            iwiSupportByItem[item] += item.myExternalUtil;
        }
    }
}

std::set<Pattern> Utils::getUnionWithHeaderEntry(const std::set<Pattern>& prefix, const std::pair<const Item, std::shared_ptr<FPNode>>& i)
{
    
    std::set<Pattern> result = prefix;
    std::set<Item> convertedToSet = {i.first};
    Pattern toInsert = std::make_pair(convertedToSet, i.second->frequency);
    result.insert(toInsert);
    return result;
}

std::set<Pattern> Utils::IWIMining(const FPTree& fptree, const int& minSup, int level)
{
    if ( fptree.empty() ) { return {}; }

    if ( contains_single_path( fptree ) ) {
        // generate all possible combinations of the items in the tree

        std::set<Pattern> single_path_patterns;

        // for each node in the tree
        assert( fptree.root->children.size() == 1 );
        auto curr_fpnode = fptree.root->children.front();
        while ( curr_fpnode ) 
        {
            const Item& curr_fpnode_item = curr_fpnode->item;
            const int curr_fpnode_frequency = curr_fpnode->frequency;

            // add a pattern formed only by the item of the current node
            Pattern new_pattern{ { curr_fpnode_item }, curr_fpnode_frequency };
            single_path_patterns.insert( new_pattern );

            // create a new pattern by adding the item of the current node to each pattern generated until now
            for ( const Pattern& pattern : single_path_patterns ) {
                Pattern new_pattern{ pattern };
                new_pattern.first.insert( curr_fpnode_item );
                assert( curr_fpnode_frequency <= pattern.second );
                new_pattern.second = curr_fpnode_frequency;

                single_path_patterns.insert( new_pattern );
            }

            // advance to the next node until the end of the tree
            assert( curr_fpnode->children.size() <= 1 );
            if ( curr_fpnode->children.size() == 1 ) { curr_fpnode = curr_fpnode->children.front(); }
            else { curr_fpnode = nullptr; }
        }

        return single_path_patterns;
    }
    else 
    {
        // generate conditional fptrees for each different item in the fptree, then join the results

        std::set<Pattern> F;
        int count = 0;
            // for each item in the fptree
        for ( const auto& pair : fptree.header_table ) 
        {
            if (level == 0)
            {
                showProgress((float)count/(float)fptree.header_table.size());
                // std::cout << pair.first.myName << " " << count << "/" << fptree.header_table.size() << std::endl;
            }
            
            const Item& curr_item = pair.first;

            // start by generating the conditional pattern base
            std::vector<TransformedPrefixPath> conditional_pattern_base;

            // for each path in the header_table (relative to the current item)
            auto path_starting_fpnode = pair.second;
            while ( path_starting_fpnode )
            {
                // construct the transformed prefix path

                // each item in the transformed prefix path has the same frequency (the frequency of path_starting_fpnode)
                const int path_starting_fpnode_frequency = path_starting_fpnode->frequency;

                auto curr_path_fpnode = path_starting_fpnode->parent.lock();
                // check if curr_path_fpnode is already the root of the fptree
                if ( curr_path_fpnode->parent.lock() ) 
                {
                    // the path has at least one node (excluding the starting node and the root)
                    TransformedPrefixPath transformed_prefix_path{ {}, path_starting_fpnode_frequency };

                    while ( curr_path_fpnode->parent.lock() ) 
                    {
                        // assert( curr_path_fpnode->frequency >= path_starting_fpnode_frequency );
                        transformed_prefix_path.first.push_back( curr_path_fpnode->item );

                        // advance to the next node in the path
                        curr_path_fpnode = curr_path_fpnode->parent.lock();
                    }

                    conditional_pattern_base.push_back( transformed_prefix_path );
                }

                // advance to the next path
                path_starting_fpnode = path_starting_fpnode->node_link;
            }

            // generate the transactions that represent the conditional pattern base
            std::vector<EquivTrans> conditional_fptree_transactions;
            for ( const TransformedPrefixPath& transformed_prefix_path : conditional_pattern_base ) 
            {
                const std::vector<Item>& transformed_prefix_path_items = transformed_prefix_path.first;
                const int transformed_prefix_path_items_frequency = transformed_prefix_path.second;

                Transaction transaction = transformed_prefix_path_items;

                EquivTrans equivTrans(transaction, transformed_prefix_path_items_frequency);

                // add the same transaction transformed_prefix_path_items_frequency times
                // for ( auto i = 0; i < transformed_prefix_path_items_frequency; ++i ) {
                conditional_fptree_transactions.push_back( equivTrans );
                // }
            }

            // build the conditional fptree relative to the current item with the transactions just generated
            FPTree conditional_fptree( conditional_fptree_transactions, fptree.minimum_support_threshold );
            // call recursively fptree_growth on the conditional fptree (empty fptree: no patterns)
            // std::cout << "===================== BEFORE =====================" << std::endl;
            // conditional_fptree.printTree();
            conditional_fptree.pruneItems();
            // std::cout << "===================== AFTER =====================" << std::endl;
            // conditional_fptree.printTree();
            // if (!conditional_fptree.empty())
            // {
            std::set<Pattern> conditional_patterns = IWIMining(conditional_fptree, minSup, level+1);
            // }
            

            // construct patterns relative to the current item using both the current item and the conditional patterns
            std::set<Pattern> curr_item_patterns;


            // the first pattern is made only by the current item
            // compute the frequency of this pattern by summing the frequency of the nodes which have the same item (follow the node links)
            int curr_item_frequency = 0.00000000000000000000000000000;
            auto fpnode = pair.second;
            while ( fpnode ) 
            {
                curr_item_frequency += fpnode->frequency;
                fpnode = fpnode->node_link;
            }
            // add the pattern as a result
            if (curr_item_frequency >= minSup)
            {
                Pattern pattern{ { curr_item }, curr_item_frequency };
                curr_item_patterns.insert( pattern );
                // the next patterns are generated by adding the current item to each conditional pattern
                for ( const Pattern& pattern : conditional_patterns ) 
                {
                    Pattern new_pattern{ pattern };
                    new_pattern.first.insert( curr_item );
                    if (curr_item_frequency < pattern.second)
                    {
                        std::cout <<  std::setprecision(15) << "Precision error" << pattern.second << "||" << curr_item_frequency << std::endl;
                    }
                    assert((int) curr_item_frequency >= (int) pattern.second );
                    new_pattern.second = pattern.second;
                    
                    if(calcDiversification(new_pattern.first) > minDiv)
                    {
                        curr_item_patterns.insert( { new_pattern } );
                    }

                }

                // join the patterns generated by the current item with all the other items of the fptree
                F.insert( curr_item_patterns.cbegin(), curr_item_patterns.cend() );
            }
            

            count++;
        }
        return F;

    // return multi_path_patterns;
    }
}

std::map<Item, int>& Utils::getIwiSupportByItem()
{
    return iwiSupportByItem;
}

void Utils::printEquivTrans(const std::list<EquivTrans>& equivTransList)
{
    std::cout << "==================EQUIV TRANS==================" << std::endl;
    for(const auto& equivTrans : equivTransList )
    {
        for (const auto& item : equivTrans.myItems)
        {
            std::cout << item.myName << " ";
        }
        std::cout << ": " << equivTrans.myWeight << std::endl;
    }
    std::cout << "================================================" << std::endl;

}

int Utils::calcDiversification(std::set<Item> items)
{
    std::unordered_set<int> sectorSet;
    for (const auto& item : items)
    {
        sectorSet.insert(item.mySector);
    }

    return ( (float) sectorSet.size()/(float)items.size() )*100;
}

void Utils::showProgress(const float& progress)
{
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) 
    {
        if (i < pos) 
        {
            std::cout << "=";
        }
        else if (i == pos) 
        {
            std::cout << ">";
        }
        else
        {
            std::cout << " ";
        } 
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();

    if(progress == 1)
    {
        std::cout << std::endl;
    }
}


Utils::~Utils() {

}