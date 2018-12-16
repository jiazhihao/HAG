#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <unordered_map>
#include <set>
#include <assert.h>

namespace std{
  template<>
  struct hash<std::pair<int, int> >
  {
    size_t operator()(const std::pair<int, int>& p) const
    {
      size_t res = 17;
      res = res * 31 + hash<int>()(p.first);
      res = res * 31 + hash<int>()(p.second);
      return res;
    }
  };
}

struct PairCount {
  PairCount(int u, int v, int _cnt) : fst(u), snd(v), cnt(_cnt) {}
  int fst, snd, cnt;
};

struct pair_count_compare {
  bool operator()(const PairCount& lhs, const PairCount& rhs) const {
    if (lhs.cnt != rhs.cnt) return (lhs.cnt > rhs.cnt);
    if (lhs.fst != rhs.fst) return (lhs.fst < rhs.fst);
    return (lhs.snd < rhs.snd);
  }
};

void add_pair_count(int u, int v,
                    std::unordered_map<std::pair<int, int>, int>& counter,
                    std::set<PairCount, pair_count_compare>& heap)
{
  if (u > v) {int w = u; u = v; v = w;}
  if (counter.find(std::make_pair(u, v)) == counter.end()) {
    counter[std::make_pair(u, v)] = 1;
    PairCount pc(u, v, 1);
    heap.insert(pc);
  } else {
    int oldVal = counter[std::make_pair(u, v)];
    PairCount pc(u, v, oldVal);
    heap.erase(pc);
    counter[std::make_pair(u, v)] = oldVal + 1;
    pc.cnt = oldVal + 1;
    heap.insert(pc);
  }
}

void sub_pair_count(int u, int v,
                    std::unordered_map<std::pair<int, int>, int>& counter,
                    std::set<PairCount, pair_count_compare>& heap)
{
  if (u > v) {int w = u; u = v; v = w;}
  int oldVal = counter[std::make_pair(u, v)];
  PairCount pc(u, v, oldVal);
  heap.erase(pc);
  counter[std::make_pair(u, v)] = oldVal - 1;
  pc.cnt = oldVal - 1;
  heap.insert(pc);
}

int main()
{
  //FILE* file = fopen("IMDB-MULTI/IMDB-MULTI_A.txt", "r");
  //FILE* file = fopen("REDDIT-BINARY/REDDIT-BINARY_A.txt", "r");
  //FILE* file = fopen("PROTEINS_full/PROTEINS_full_A.txt", "r");
  //FILE* file = fopen("COLLAB/COLLAB_A.txt", "r");
  FILE* file = fopen("BZR_MD/BZR_MD_A.txt", "r");
  int u, v;
  int nv = 0;
  std::map<int, std::set<int>* > inEdges;
  std::unordered_map<std::pair<int, int> , int> counter;
  std::set<PairCount, pair_count_compare> heap;

  while (fscanf(file, "%d, %d", &u, &v) != EOF) {
    if (std::max(u, v) >= nv)
      nv = std::max(u, v) + 1;
    if (inEdges.find(v) == inEdges.end())
      inEdges[v] = new std::set<int>();
    else
      inEdges[v]->insert(u);
  }
  fclose(file);
  printf("nv = %d\n", nv);
  for (int i = 0; i< nv; i++)
    if (inEdges.find(i) != inEdges.end()) {
      std::set<int>::const_iterator it1, it2;
      std::set<int>::const_iterator first = inEdges[i]->begin(), last = inEdges[i]->end();
      for (it1 = first; it1 != last; it1 ++)
        for (it2 = first; it2 != it1; it2 ++) {
          u = *it2;
          v = *it1;
          assert(u < v);
          if (counter.find(std::make_pair(u, v)) == counter.end())
            counter[std::make_pair(u, v)] = 1;
          else
            counter[std::make_pair(u, v)] ++;
        }
      if (i % 1000 == 0) printf("i = %d\n", i);
    }

  // initialize heap
  std::unordered_map<std::pair<int, int>, int>::const_iterator it;
  for (it = counter.begin(); it != counter.end(); it++) {
    PairCount pc(it->first.first, it->first.second, it->second);
    heap.insert(pc);
  }
  
  int saved = 0;
  std::map<int, int> depths;
  for (int i = 0;; i++) {
    PairCount pc = *heap.begin();
    int preDepth = 0;
    if (depths.find(pc.fst) != depths.end())
      preDepth = std::max(preDepth, depths[pc.fst]);
    if (depths.find(pc.snd) != depths.end())
      preDepth = std::max(preDepth, depths[pc.snd]);
    depths[nv] = preDepth + 1;
    saved += pc.cnt;
    printf("pc[%d]: fst(%d) snd(%d) depth(%d) cnt(%d) acc_save(%d)\n", i, pc.fst, pc.snd, preDepth + 1, pc.cnt, saved);
    if (pc.cnt < 3) break;
    heap.erase(heap.begin());
    for (int j = 0; j < nv; j++)
      if (inEdges.find(j) != inEdges.end()) {
        std::set<int>* list = inEdges[j];
        if ((list->find(pc.fst) != list->end())
        &&  (list->find(pc.snd) != list->end())) {
          list->erase(pc.fst);
          list->erase(pc.snd);
          // update counters
          std::set<int>::const_iterator it;
          for (it = list->begin(); it != list->end(); it++) {
            sub_pair_count(*it, pc.fst, counter, heap);
            sub_pair_count(*it, pc.snd, counter, heap);
            add_pair_count(*it, nv, counter, heap);
          }
          list->insert(nv);
        }
      }
    nv ++;
  }
}

