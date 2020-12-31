#include <iostream>
#include <array>
#include <deque>
#include <cstring>
#include <string>
#include <vector>

#define TECA_ERROR(_msg) \
    std::cerr << "" _msg << std::endl;

// split block 1 into 2 blocks in the d direction. block1 is modified in place
// and the new block is returned in block 2. return 1 if the split succeeded
// and 0 if it failed.
int split(std::array<unsigned long,6> &block_1, std::array<unsigned long,6> &block_2, int d)
{
    // compute length in this direction
    int i0 = 2*d;
    int i1 = i0 + 1;

    unsigned long ni = block_1[i1] - block_1[i0] + 1;

    // can't split in this direction
    if (ni < 2)
        return 0;

    // compute the new length
    unsigned long no = ni/2;

    // copy input
    block_2 = block_1;

    // split
    block_1[i1] = block_1[i0] + no;
    block_2[i0] = block_1[i1] + 1;

    return 1;
}



int partition(const std::array<unsigned long,6> &ext, int n_blocks, std::deque<std::array<unsigned long, 6>> &blocks)
{
    // get the length in each direction
    unsigned long nx = ext[1] - ext[0] + 1;
    unsigned long ny = ext[3] - ext[2] + 1;
    unsigned long nz = ext[5] - ext[4] + 1;
    unsigned long nxyz = nx*ny*nz;

    // check that it is possible to generate the requested number of blocks
    if (nxyz < n_blocks)
    {
        TECA_ERROR("Can't split " << nxyz << " cells " << " into " << n_blocks)
        return -1;
    }

    // which directions can we split in?
    std::vector<int> dirs;
    if (nx > 1)
        dirs.push_back(0);
    if (ny > 1)
        dirs.push_back(1);
    if (nz > 1)
        dirs.push_back(2);

    int n_dirs = dirs.size();

    // start with the full extent
    blocks.push_back(ext);

    // split each block until the desired number is reached.
    while (blocks.size() < n_blocks)
    {
        // alternate splitable directions
        for (int d = 0; d < n_dirs; ++d)
        {
            // make a pass overt each block split it into 2 until the
            // desired number is realized
            unsigned long n = blocks.size();
            for (unsigned long i = 0; i < n; ++i)
            {
                // take the next block from the front
                std::array<unsigned long,6> b2;
                std::array<unsigned long,6> b1 = blocks.front();
                blocks.pop_front();

                // add the new blocks to the back
                if (split(b1, b2, d))
                    blocks.push_back(b2);
                blocks.push_back(b1);

                // are we there yet?
                if (blocks.size() == n_blocks)
                    return 0;
            }
        }
    }

    return 0;
}


std::ostream &operator<<(std::ostream &os, const std::array<unsigned long,6> &blk)
{
    os << blk[0] << ", " << blk[1] <<  ", " << blk[2] << ", " << blk[3] << ", " << blk[4] << ", " << blk[5];
    return os;
}


int main(int argc, char **argv)
{
    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int nz = atoi(argv[3]);
    int n_blks = atoi(argv[4]);

    std::array<unsigned long,6> ext({0,nx-1, 0,ny-1, 0,nz-1});

    std::cerr << "ext = " << ext << std::endl;

    std::deque<std::array<unsigned long,6>> blocks;

    if (partition(ext, n_blks, blocks))
    {
        TECA_ERROR("partition failed!")
        return -1;
    }

    int nb = blocks.size();
    for (int i = 0; i < nb; ++i)
    {
        std::cerr << i << "    " << blocks[i] << std::endl;
    }

    return 0;
}


