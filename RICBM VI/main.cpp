// testing hello owen
// ============================================================================
//  SEA DRAGON  -  UNSW Battlecode "sea dragons" bot  (C++20, one file)
// ============================================================================
//
//  HOW TO SUBMIT
//    1. unswbc init cpp mybot          (creates the project folder)
//    2. Replace mybot/main.cpp with this file.  No helper header is used:
//       this file speaks the engine's wire protocol (v2.1.0) itself.
//    3. mybot/bot.toml:
//           [project]
//           language = "cpp"
//           include = [ "main.cpp" ]
//       (if the build complains, keep the bot.toml that `init` generated
//        and just make sure main.cpp is in its include list)
//    4. unswbc submit mybot
//
//  RUNTIME BUDGET
//    The judge compiles with clang -O2 -std=c++20 -msimd128 (libc++,
//    wasi-libc) and charges CPU points: 100M per dragon per turn, and its
//    clock advances 1 ns per point.  A normal turn here costs a few million
//    points; the heavier searches (escape sprints, split checks) stop at
//    ~45M.  stdout is fully buffered and written once per turn (every write
//    costs 2.5M points).
//
//  STRATEGY
//  --------
//  Win conditions, in order: eliminate every enemy dragon; else, after round
//  500, own the single longest living dragon; else the larger total length.
//
//  1. Never die for free.  Every candidate action is replayed step by step
//     with the engine's exact rules (own body checked before the tail moves,
//     sprint payment, head-to-head, portals) and scored with a time-aware
//     flood fill: segment i of a length-L dragon frees up L-i+1 moves from
//     now, and every pearl we may be forced to eat on the way delays our own
//     tail by one more move (a pocket full of pearls is smaller than it
//     looks).  Trapped dragons escape by sprinting, or by splitting so the
//     long rear half leaves as a new dragon through the old tail.
//
//  2. Grow fast.  Pearls are foraged with a discounted "pearl field" over
//     BFS distances from each candidate head position, weighted by how likely
//     a pearl is there (seen now, remembered, or predicted from the tile's
//     countdown) and discounted when another head is closer (Voronoi
//     competition, which also stops teammates chasing the same pearl and
//     colliding).
//
//  3. Multiply fast.  A dragon splits into halves as soon as it is 4 long,
//     until the team reaches a cap set by the arena's size, openness and
//     pearl richness (up to 16 on tiny arenas, up to 64 on big open ones).
//     More dragons = more pearls per round, more trades, and no single
//     missile can eliminate us.  A few slots are kept free so that emergency
//     splits stay legal at the cap.
//
//  4. Protect length.  A dragon is worth more than its length once it is
//     long (it is the likely tiebreak winner), so long dragons refuse risky
//     pearls and trades, and price every tile an enemy head could sprint into
//     this turn.  If a strike on a long dragon cannot be dodged it "decoy
//     splits": its rear bulk leaves from the old tail and the enemy can only
//     hit a 2-segment front.  The first dragon (anchor) seeds the swarm with
//     a couple of splits, then only grows.
//
//  5. Missiles.  A head-to-head kills both dragons, and a dragon of length L
//     can move up to L-1 tiles in one turn, so small dragons sprint into the
//     heads of bigger enemies (the enemy's longest dragon is their tiebreak).
//     Bodyguards strike enemies that come near a big friendly head.  How
//     often the enemy strikes is learned from what we see (sprints, strikes,
//     chances they passed up) and scales our caution.
//
//  6. Arena memory.  Each dragon remembers every edge (kelp / portal) and
//     tile (spawn class, countdown, pearl) it has seen, detects the map
//     symmetry (x-mirror, y-mirror or 180 rotation) from contradictions
//     between mirrored observations, then fills in the mirror half for free.
//     Portals are paired by their shared id; BFS walks through paired portals
//     exactly as the engine does (enter on one side of an edge, leave on the
//     same side of its partner, direction kept).  Portals whose exit cannot
//     be seen are only taken when the expected gain beats the risk.
//
//  7. Sonar.  Every turn each dragon casts one authenticated 32-bit message
//     (3-bit type, 21-bit payload, 8-bit keyed MAC of type, payload, round
//     and team, so enemy chatter and replays are rejected).  Payloads: map
//     symmetry + enemy aggression, portal endpoints, kelp runs, enemy-head
//     sightings and the position of a big friendly dragon.  Received facts
//     never override what a dragon sees itself, so a forged message cannot
//     kill us.
//
//  FILE MAP
//    Params ........ tunable constants (values chosen by simulation)
//    Parser ........ line-driven state machine for the engine's round block
//    Arena memory .. edges, portals, spawn timers, symmetry, own-body trail
//    World model ... occupancy with free-up times, other dragons, threat map
//    Search ........ timed flood fill, Voronoi fill, exact step replay
//    Policy ........ team-size cap, split rules, move scoring, sprints
//    Sonar ......... message encoding / decoding
//    Turn .......... act(): observe -> model -> candidates -> decide -> reply
//    main() ........ one process per dragon; exits when stdin closes
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace seadragon {

using u32 = uint32_t;
using u64 = uint64_t;

// ---------------------------------------------------------------------------
// Directions: DIR_N=0 DIR_E=1 DIR_S=2 DIR_W=3 (opposite = d ^ 2)
// ---------------------------------------------------------------------------
enum : int { DIR_N = 0, DIR_E = 1, DIR_S = 2, DIR_W = 3 };
static constexpr int DX[4] = {0, 1, 0, -1};
static constexpr int DY[4] = {-1, 0, 1, 0};
static constexpr char DCH[4] = {'N', 'E', 'S', 'W'};
static constexpr int INF = 1 << 28;

// Edge kinds as remembered by a dragon
static constexpr int8_t K_UNK = -1, K_OPEN = 0, K_KELP = 1, K_PORT = 2;

static inline u32 mix32(u32 x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
static inline long long nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A dragon (other than us) as reconstructed from this turn's window.
struct Other {
  int id = -1, team = -1;
  int head = -1;            // head tile, -1 if head is outside the window
  int face = 0;             // heading
  std::vector<int> chain;   // tiles from the head backwards (chain[0] = head)
  int visible = 0;          // segments of this dragon inside the window
  bool tailSeen = false;    // the chain provably ends at the tail
  int lenLow = 0;           // lower bound on its length
  int lenKnown = -1;        // exact length when the whole body is visible
  bool enemy = false;
};

// One way of acting this turn.
struct Cand {
  int kind = 0;               // 0 move, 1 split
  std::string steps;          // for moves
  int split = 0;              // for splits
  std::vector<int> path;      // tiles visited (moves)
  int dest = -1;              // final head tile
  bool ate = false;
  int pearlsOnPath = 0;
  bool attack = false;        // ends in an enemy head (both die)
  int attackId = -1;
  bool unknownExit = false;   // crosses a portal whose exit is not visible
  double score = -1e18;
  const char* tag = "";
  int space = -1, terr = -1, vor = -1;
  double field = 0, courtesy = 0;
};

// ---------------------------------------------------------------------------
// Tunable strategy constants (values chosen by simulation, see header).
// ---------------------------------------------------------------------------
struct Params {
  double capDiv = 6.0;       // arena tiles per dragon when sizing the swarm
  int capMin = 2;            // never fewer dragons than this (elimination insurance)
  int capSmallMax = 16;      // cap on tiny arenas (<= 170 tiles)
  int splitLen = 4;          // split when at least this long (child = half)
  int splitLenSmall = 4;     // ... on arenas of <= 300 tiles
  int anchorSeed = 3;        // the anchor splits until the team has this many
  int anchorSeedRound = 80;  // ... and only before this round
  double unitK = 240.0;      // worth of a unit = unitK / unitCount (+ length)
  double enemyUnit = 10.0;   // assumed worth of an enemy unit
  double strikeGain = 0.8;   // how much we value an enemy loss vs our own
  double pAdj = 0.9, pFar = 0.7;
  double wField = 6.0, wEat = 2.0, wTrap = 220.0, wSpace = 1.2, wTerr = 3.0, wRisk = 1.0,
         wLen = 3.0, wContest = 45.0;
  double gamma = 0.82;
  int lateNoSplit = 440;
  int growLen = 12;          // dragons this long stop splitting and grow (tiebreak)
  double aggrPrior = 0.7;    // prior probability an enemy strikes when it can
  double huntW = 1.2;        // pull towards enemy heads worth striking
  double tinySprint = 3.0;   // extra cost of sprinting when it leaves us <= 3 long
  double pDecay = 0.12;      // strike probability drops by this per extra tile of path
  double threatSprint = 8.0; // consider escape sprints when the best move carries this much threat
  int sprintMinLen = 12;     // ... but only for dragons at least this long (or the anchor)
  int ghostUnits = 4;        // remember recently seen enemy heads while we have <= this many dragons
  int ghostAge = 4;          // ... for this many rounds
  double ghostDecay = 0.6;   // threat scale per round of age
  double bigW = 1.0;         // extra worth per segment of a big dragon (tiebreak candidate)
  int bigFrom = 8;           // ... ramping in from this length
  int bigFull = 24;          // ... to full weight at this length
  int reserve = 3;           // unit slots kept free for emergency (escape / decoy) splits
  int reserveSmall = 2;      // emergency splits of dragons shorter than decoyLen leave this many slots
  int decoyLen = 8;          // dragons at least this long may split off their bulk to dodge a strike
  double decoyThr = 4.0;     // ... when the chosen move carries at least this expected loss
  int pearlDelayCap = 6;     // pocket check: pearls we may be forced to eat delay our tail by this much at most
};

class Bot {
 public:
  Params P;
  // Feed one line of engine input. Returns true when a full round block has
  // been read and a reply must be produced with act().
  bool feed(char* line);
  std::string act();
  bool finished = false;

  // ------------------------------------------------------------------ identity
  int myId = -1, team = 0, W = 0, H = 0, NT = 0, unitLimit = 64;
  bool mapReady = false;
  u32 salt = 0;

  // ---------------------------------------------------------- raw turn input
  int round = 0, facing = 0, length = 0, unitCount = 1;
  std::vector<u32> inbox;
  int wx[49], wy[49], wp[49], wc[49];
  struct RawSeg { int team, id, x, y, face, head; };
  std::vector<RawSeg> rsegs;
  int hK[8][7], hP[8][7], vK[7][8], vP[7][8];

  // ------------------------------------------------------------ parser state
  enum { PH_IDLE, PH_MSGS, PH_TILES, PH_BODYHDR, PH_BODIES, PH_HE, PH_VE };
  int phase = PH_IDLE, left = 0, row = 0;
  bool inRound = false;

  // ------------------------------------------------------------ arena memory
  std::vector<int8_t> ek[2];     // [0] north edge of tile, [1] west edge of tile
  std::vector<int> ep[2];        // portal id (or -1 when only known via mirror)
  std::vector<int> epart[2];     // partner tile of a portal edge (same orientation)
  std::vector<int16_t> seenR;    // round a tile was last inside our window
  std::vector<int8_t> spawnK;    // -1 unknown, 0 never spawns, 1 spawns
  std::vector<int16_t> cdR, cdV; // countdown observation: value cdV at round cdR
  std::vector<int16_t> pearlR;   // last round seen WITH a pearl (-1)
  std::vector<int16_t> emptyR;   // last round seen WITHOUT a pearl (-1)
  std::vector<int16_t> occR;     // last round seen occupied by a dragon (-1)
  // Our own rear segments we have seen but whose place in the body we do not
  // know (a split child starts knowing only what is in its window).  A body
  // tile stays occupied until the tail passes it, so it can be remembered:
  // the tile is blocked until round ownMemUntil (the unknown rear shrinks by
  // one segment per move).
  std::vector<int16_t> ownMemUntil;
  std::vector<int> ownMemList;
  struct PObs { int id, o, t; };
  std::vector<PObs> pobs;        // every portal edge we know, with its id
  struct PClaim { int hash, o, t; };
  std::vector<PClaim> pclaims;   // portal ends heard over sonar (id hashed)

  // symmetry detection: candidates 1 = x mirror, 2 = y mirror, 3 = rotation
  int symKnown = 0;
  bool cdShared = false;         // mirrored tiles share countdowns (as documented)
  int gBad[4] = {0, 0, 0, 0}, gGood[4] = {0, 0, 0, 0};
  int cBad[4] = {0, 0, 0, 0}, cGood[4] = {0, 0, 0, 0};
  bool symHeard = false;

  // ---------------------------------------------------------------- own body
  std::vector<int> trail;        // our body, head first (known prefix)
  int bodyKnown = 0;
  int firstRound = -1;           // round of the first block this process saw
  bool isAnchor = false;
  int startTile = -1;
  int lastKind = -1;             // what we did last turn
  std::vector<int> lastPath;
  int lastSplit = 0;
  int expectHead = -1;
  int turnsAlive = 0;
  int lastTarget = -1;

  // --------------------------------------------------------- per-turn scratch
  int headT = -1;
  std::vector<int> nbr;          // nbr[t*4+d]: destination tile, -1 kelp, -2 unknown portal
  bool nbrAll = true;            // neighbour table must be rebuilt from scratch
  std::vector<int> nbrDirty;     // tiles whose neighbour entries must be refreshed
  void dirtyEdge(int o, int t) {  // both tiles touching edge (o,t)
    nbrDirty.push_back(t);
    nbrDirty.push_back(nbPlain(t, o == 0 ? DIR_N : DIR_W));
  }
  std::vector<int> occ;          // dragon id on tile or -1
  std::vector<int8_t> occTeam, occHead, occOwn;
  std::vector<int> blockT;       // first absolute step at which the tile can be entered
  std::vector<int> winStamp;     // == turnStamp when the tile is inside our window
  int turnStamp = 0;
  std::vector<Other> others;
  std::vector<int> dOther;       // BFS distance from the nearest other head
  std::vector<int> dEnemy;       // BFS distance from the nearest enemy head
  std::vector<float> threat;     // expected loss factor if our head ends here
  float threatHere = 0.f;        // same, for staying where we are (splitting)
  float threatTail = 0.f;        // same, for a child born on our tail
  std::vector<float> pval;       // pearl value per tile
  std::vector<int8_t> futureCd;  // visible empty spawn tile: rounds until spawn
  std::vector<int> bfsQ, dist, dist2;
  std::vector<float> gpow;
  long long t0 = 0;
  double density = 0;            // fraction of window tiles occupied by dragons
  double kelpFrac = 0;           // fraction of known edges that are kelp
  int knownEdges = 0, knownKelp = 0;
  int spawnKnown = 0, spawnYes = 0;
  double gapSum = 0; int gapCnt = 0;
  double aggrPass = 0, aggrHit = 0;   // evidence: missed strike chances / sprints seen
  double aggrHeard = -1;              // aggression estimate heard over sonar
  // Real evidence that the opponent trades head-on (sprints seen, strikes
  // witnessed, or a teammate reporting a high estimate over sonar).
  bool kamikazeEvidence() const { return aggrHit >= 1.0 || aggrHeard >= 0.6; }
  double aggression() const {
    double a0 = 3.0 * P.aggrPrior, b0 = 3.0 * (1.0 - P.aggrPrior);
    double e = (a0 + aggrHit) / (a0 + b0 + aggrHit + aggrPass);
    if (aggrHeard >= 0 && aggrPass + aggrHit < 4) e = 0.5 * (e + aggrHeard);
    return std::clamp(e, 0.05, 0.95);
  }
  struct PrevEnemy { int id, head, len; };
  std::vector<PrevEnemy> prevEnemies;
  struct Seen { int id, team, head, len; };
  std::vector<Seen> lastSeen;        // dragons with a visible head last turn
  struct Ghost { int id, head, len, reach, r; };
  std::vector<Ghost> ghosts;         // enemy heads seen recently, now out of sight
  std::vector<std::pair<int, int>> lastEnemyHeads;  // (id, tile) last turn
  struct Sight { int t, len, r; };
  std::vector<Sight> sightings;  // enemy heads reported over sonar
  int friendBigT = -1, friendBigLen = 0, friendBigR = -1;
  int msgCursor = 0;

  // ---------------------------------------------------------------- helpers
  inline int tileOf(int x, int y) const { return y * W + x; }
  inline int nbPlain(int t, int d) const {
    int x = t % W, y = t / W;
    x += DX[d]; y += DY[d];
    if (x < 0) x += W; else if (x >= W) x -= W;
    if (y < 0) y += H; else if (y >= H) y -= H;
    return y * W + x;
  }
  // The edge crossed when leaving tile t in direction d, as (orientation, tile).
  inline void edgeOf(int t, int d, int& o, int& et) const {
    switch (d) {
      case DIR_N: o = 0; et = t; break;
      case DIR_S: o = 0; et = nbPlain(t, DIR_S); break;
      case DIR_W: o = 1; et = t; break;
      default: o = 1; et = nbPlain(t, DIR_E); break;
    }
  }
  // Exit tile when crossing the partner portal edge p (same orientation) in d.
  inline int portalExit(int p, int d) const {
    // enter from the left -> emerge on the right of the partner edge, etc.
    switch (d) {
      case DIR_N: return nbPlain(p, DIR_N);
      case DIR_S: return p;
      case DIR_W: return nbPlain(p, DIR_W);
      default: return p;
    }
  }
  // Destination of one step using memory. -1 kelp, -2 portal with unknown exit.
  int stepMem(int t, int d) const {
    int o, et; edgeOf(t, d, o, et);
    int8_t k = ek[o][et];
    if (k == K_KELP) return -1;
    if (k == K_PORT) {
      int p = epart[o][et];
      return p < 0 ? -2 : portalExit(p, d);
    }
    return nbPlain(t, d);  // open or unknown (optimistic for planning)
  }
  int mirrorT(int t, int s) const {
    int x = t % W, y = t / W;
    if (s & 1) x = W - 1 - x;
    if (s & 2) y = H - 1 - y;
    return y * W + x;
  }
  int mirrorE(int o, int t, int s) const {
    int x = t % W, y = t / W;
    if (o == 0) { if (s & 1) x = W - 1 - x; if (s & 2) y = (H - y) % H; }
    else        { if (s & 1) x = (W - x) % W; if (s & 2) y = H - 1 - y; }
    return y * W + x;
  }
  int chebyshev(int a, int b) const {
    int dx = std::abs(a % W - b % W), dy = std::abs(a / W - b / W);
    dx = std::min(dx, W - dx); dy = std::min(dy, H - dy);
    return std::max(dx, dy);
  }
  int manhattan(int a, int b) const {
    int dx = std::abs(a % W - b % W), dy = std::abs(a / W - b / W);
    dx = std::min(dx, W - dx); dy = std::min(dy, H - dy);
    return dx + dy;
  }
  bool inWin(int t) const { return winStamp[t] == turnStamp; }

  // ---------------------------------------------------------------- stages
  void setupMap();
  void integrate();
  void recordEdge(int o, int t, int kind, int pid, bool direct);
  void registerPortal(int pid, int o, int t);
  void pairPortals(int o, int a, int b);
  void symEvidenceEdge(int o, int t, int kind);
  void symEvidenceTile(int t, int spawn, int cdv);
  void decideSymmetry();
  void applySymmetryToAll();
  void mirrorTileInfo(int t);
  void updateTrail();
  void rebuildTrailFromWindow();
  void buildNeighbors();
  void buildOccupancy();
  void buildDistances();
  void buildPearlValues();
  int timedBfs(int src, int s0, int ownShift, std::vector<int>& d, int& territory, int capCount);
  int voronoiBfs(int src, int ownShift, int enough = 1 << 30);
  double courtesyCost(int dest);
  std::vector<int> dist3;
  bool simulate(const std::string& steps, Cand& c);
  double evaluate(Cand& c);
  void genCandidates(std::vector<Cand>& out);
  int unitCap() const;
  // Worth of one of our dragons in "length units": fewer dragons -> each one
  // matters more (elimination), plus its body; the anchor carries the tiebreak.
  double lenWorth(int len) const {
    double f = std::clamp((double)(len - P.bigFrom) / std::max(1, P.bigFull - P.bigFrom), 0.0, 1.0);
    return len * (1.0 + P.bigW * f);
  }
  double myWorth() const { return P.unitK / std::max(1, unitCount) + lenWorth(length); }
  bool anchorNow() const { return isAnchor && length >= 6; }
  double enemyWorth(int len) const { return P.enemyUnit + len; }
  int splitLen() const;
  bool splitSafe(int n, double& quality);
  int childRoom(int n, int cap);
  u32 mac(u32 type, u32 payload, int r) const;
  void readSonar();
  bool chooseSonar(u32& out);
  void parseEdgeToken(const char* tok, int& kind, int& pid);
};

// ============================================================================
//  PARSER: line-driven state machine for wire protocol 2.1.0
// ============================================================================
static int splitTokens(char* s, char** tok, int maxTok) {
  int n = 0;
  while (*s && n < maxTok) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    if (!*s) break;
    tok[n++] = s;
    while (*s && *s != ' ' && *s != '\t' && *s != '\r' && *s != '\n') s++;
    if (*s) *s++ = 0;
  }
  return n;
}

void Bot::parseEdgeToken(const char* tok, int& kind, int& pid) {
  if (tok[0] == '.' && tok[1] == 0) { kind = K_OPEN; pid = -1; return; }
  const char* p = tok;
  bool num = (*p != 0);
  if (*p == '-' || *p == '+') p++;
  if (!*p) num = false;
  for (; *p; p++) if (*p < '0' || *p > '9') { num = false; break; }
  if (num) { kind = K_PORT; pid = atoi(tok); return; }
  kind = K_KELP; pid = -1;  // 'w' (and anything unexpected) is treated as kelp
}

bool Bot::feed(char* line) {
  if (char* h = strchr(line, '#')) *h = 0;  // helpers strip comments; so do we
  char* tok[16];
  int n = splitTokens(line, tok, 16);
  if (n == 0) return false;
  // Keywords are recognised in any phase, which keeps us in sync even if a
  // count were ever off by one.
  const char* k = tok[0];
  if (!strcmp(k, "ID") && n >= 2) { myId = atoi(tok[1]); return false; }
  if (!strcmp(k, "TEAM") && n >= 2) { team = (tok[1][0] == 'B') ? 1 : 0; return false; }
  if (!strcmp(k, "MAP") && n >= 3) { W = atoi(tok[1]); H = atoi(tok[2]); setupMap(); return false; }
  if (!strcmp(k, "UNIT_LIMIT") && n >= 2) { unitLimit = atoi(tok[1]); return false; }
  if (!strcmp(k, "ENDGAME")) { finished = true; return false; }
  if (!strcmp(k, "ROUND") && n >= 2) {
    round = atoi(tok[1]); inRound = true; phase = PH_IDLE;
    inbox.clear(); rsegs.clear(); return false;
  }
  if (!strcmp(k, "DIR") && n >= 2) {
    facing = (tok[1][0] == 'N') ? DIR_N : (tok[1][0] == 'E') ? DIR_E : (tok[1][0] == 'S') ? DIR_S : DIR_W;
    return false;
  }
  if (!strcmp(k, "LENGTH") && n >= 2) { length = atoi(tok[1]); return false; }
  if (!strcmp(k, "UNIT_COUNT") && n >= 2) { unitCount = atoi(tok[1]); return false; }
  if (!strcmp(k, "NUM_MSGS") && n >= 2) {
    left = atoi(tok[1]); row = 0;
    phase = left > 0 ? PH_MSGS : PH_TILES;
    return false;
  }
  if (!strcmp(k, "DRAGON_BODIES") && n >= 2) {
    left = atoi(tok[1]); row = 0;
    phase = left > 0 ? PH_BODIES : PH_HE;
    return false;
  }
  switch (phase) {
    case PH_MSGS:
      inbox.push_back((u32)strtoul(tok[0], nullptr, 10));
      if (--left <= 0) { phase = PH_TILES; row = 0; }
      return false;
    case PH_TILES:
      if (n >= 4 && row < 49) {
        wx[row] = atoi(tok[0]); wy[row] = atoi(tok[1]);
        wp[row] = atoi(tok[2]); wc[row] = atoi(tok[3]);
        if (++row >= 49) phase = PH_BODYHDR;
      }
      return false;
    case PH_BODIES:
      if (n >= 6) {
        RawSeg s;
        s.team = (tok[0][0] == 'B') ? 1 : 0; s.id = atoi(tok[1]);
        s.x = atoi(tok[2]); s.y = atoi(tok[3]);
        char f = tok[4][0];
        s.face = (f == 'N') ? DIR_N : (f == 'E') ? DIR_E : (f == 'S') ? DIR_S : DIR_W;
        s.head = atoi(tok[5]);
        rsegs.push_back(s);
      }
      if (--left <= 0) { phase = PH_HE; row = 0; }
      return false;
    case PH_HE:
      if (row < 8) {
        for (int c = 0; c < 7; c++) {
          if (c < n) parseEdgeToken(tok[c], hK[row][c], hP[row][c]);
          else { hK[row][c] = K_UNK; hP[row][c] = -1; }
        }
        if (++row >= 8) { phase = PH_VE; row = 0; }
      }
      return false;
    case PH_VE:
      if (row < 7) {
        for (int c = 0; c < 8; c++) {
          if (c < n) parseEdgeToken(tok[c], vK[row][c], vP[row][c]);
          else { vK[row][c] = K_UNK; vP[row][c] = -1; }
        }
        if (++row >= 7) {
          phase = PH_IDLE;
          if (inRound && mapReady) { inRound = false; return true; }
        }
      }
      return false;
    default:
      return false;
  }
}

// ============================================================================
//  ARENA MEMORY
// ============================================================================
void Bot::setupMap() {
  if (W <= 0 || H <= 0) return;
  NT = W * H;
  for (int o = 0; o < 2; o++) {
    ek[o].assign(NT, K_UNK); ep[o].assign(NT, -1); epart[o].assign(NT, -1);
  }
  seenR.assign(NT, -1); spawnK.assign(NT, -1);
  cdR.assign(NT, -1); cdV.assign(NT, 0);
  pearlR.assign(NT, -1); emptyR.assign(NT, -1); occR.assign(NT, -1);
  ownMemUntil.assign(NT, -1); ownMemList.clear();
  nbr.assign(NT * 4, -1);
  occ.assign(NT, -1); occTeam.assign(NT, -1); occHead.assign(NT, 0); occOwn.assign(NT, 0);
  blockT.assign(NT, 0); winStamp.assign(NT, -1);
  dOther.assign(NT, INF); dEnemy.assign(NT, INF);
  threat.assign(NT, 0.f); pval.assign(NT, 0.f); futureCd.assign(NT, -1);
  bfsQ.assign(NT + 8, 0); dist.assign(NT, INF); dist2.assign(NT, INF); dist3.assign(NT, INF);
  gpow.assign(NT + 8, 0.f);
  mapReady = true;
}

// Pair two portal edges of the same orientation.
void Bot::pairPortals(int o, int a, int b) {
  if (a == b) return;
  epart[o][a] = b; epart[o][b] = a;
  dirtyEdge(o, a); dirtyEdge(o, b);
  if (ek[o][a] == K_UNK) ek[o][a] = K_PORT;
  if (ek[o][b] == K_UNK) ek[o][b] = K_PORT;
  // A mirrored pair is also a pair.
  if (symKnown) {
    int ma = mirrorE(o, a, symKnown), mb = mirrorE(o, b, symKnown);
    if (ma != mb && epart[o][ma] != mb) {
      epart[o][ma] = mb; epart[o][mb] = ma;
      dirtyEdge(o, ma); dirtyEdge(o, mb);
      if (ek[o][ma] == K_UNK) ek[o][ma] = K_PORT;
      if (ek[o][mb] == K_UNK) ek[o][mb] = K_PORT;
    }
  }
}

// Every portal id is shared by exactly two edges: seeing both pairs them.
void Bot::registerPortal(int pid, int o, int t) {
  if (pid < 0) return;
  for (auto& p : pobs) if (p.o == o && p.t == t) { p.id = pid; goto have; }
  pobs.push_back({pid, o, t});
have:
  for (auto& p : pobs)
    if (p.id == pid && !(p.o == o && p.t == t) && p.o == o) pairPortals(o, p.t, t);
  // a sonar claim with the same (hashed) id at a different edge also pairs
  for (auto& c : pclaims)
    if (c.o == o && c.t != t && c.hash == (pid & 0xFF) && epart[o][t] < 0) pairPortals(o, c.t, t);
}

// Symmetry evidence from an edge seen for the first time.
void Bot::symEvidenceEdge(int o, int t, int kind) {
  for (int s = 1; s <= 3; s++) {
    int m = mirrorE(o, t, s);
    int8_t mk = ek[o][m];
    if (mk == K_UNK) continue;
    if (m == t) continue;
    if (mk != kind) gBad[s]++;
    else if (kind != K_OPEN) gGood[s] += 2;
    else gGood[s] += 0;  // open/open is weak evidence; counted via cGood instead
  }
}

void Bot::symEvidenceTile(int t, int spawn, int cdv) {
  for (int s = 1; s <= 3; s++) {
    int m = mirrorT(t, s);
    if (m == t) continue;
    if (spawnK[m] >= 0 && seenR[m] >= 0) {
      if (spawnK[m] != spawn) gBad[s]++;
      else if (spawn == 0) gGood[s]++;
    }
    // shared countdowns: compare timelines when no reset can have happened
    if (spawn == 1 && cdR[m] >= 0) {
      int dt = round - cdR[m];
      if (dt >= 0 && dt < cdV[m]) {
        if (cdV[m] - dt == cdv) cGood[s]++;
        else cBad[s]++;
      }
    }
  }
}

void Bot::decideSymmetry() {
  if (symKnown) return;
  int alive = 0, pick = 0;
  for (int s = 1; s <= 3; s++) if (gBad[s] == 0) { alive++; pick = s; }
  if (alive == 1 && (gGood[pick] >= 4 || cGood[pick] >= 3)) { symKnown = pick; }
  else if (alive >= 2) {
    // Countdowns separate geometrically-identical candidates.
    int best = 0, cnt = 0;
    for (int s = 1; s <= 3; s++)
      if (gBad[s] == 0 && cBad[s] == 0 && cGood[s] >= 3) { best = s; cnt++; }
    int others = 0;
    for (int s = 1; s <= 3; s++) if (gBad[s] == 0 && s != best && cBad[s] >= 2) others++;
    if (cnt == 1 && others == alive - 1) symKnown = best;
  }
  if (symKnown) {
    cdShared = (cGood[symKnown] >= 1 && cBad[symKnown] == 0);
    applySymmetryToAll();
  }
}

// Copy what we know about tile t onto its mirror image.
void Bot::mirrorTileInfo(int t) {
  if (!symKnown) return;
  int m = mirrorT(t, symKnown);
  if (m == t) return;
  if (spawnK[m] < 0 && spawnK[t] >= 0) spawnK[m] = spawnK[t];
  if (cdShared && cdR[t] > cdR[m]) { cdR[m] = cdR[t]; cdV[m] = cdV[t]; }
}

void Bot::applySymmetryToAll() {
  nbrAll = true;
  for (int o = 0; o < 2; o++)
    for (int t = 0; t < NT; t++) {
      if (ek[o][t] == K_UNK) continue;
      int m = mirrorE(o, t, symKnown);
      if (ek[o][m] == K_UNK) { ek[o][m] = ek[o][t]; if (ek[o][t] != K_PORT) ep[o][m] = -1; }
    }
  for (int t = 0; t < NT; t++) mirrorTileInfo(t);
  for (int o = 0; o < 2; o++)
    for (int t = 0; t < NT; t++)
      if (epart[o][t] >= 0) {
        int a = mirrorE(o, t, symKnown), b = mirrorE(o, epart[o][t], symKnown);
        if (a != b && epart[o][a] < 0) { epart[o][a] = b; epart[o][b] = a; ek[o][a] = K_PORT; ek[o][b] = K_PORT; }
      }
}

// Store an edge observation. direct = seen with our own eyes (always wins).
void Bot::recordEdge(int o, int t, int kind, int pid, bool direct) {
  int8_t cur = ek[o][t];
  if (cur == K_UNK) {
    if (direct) symEvidenceEdge(o, t, kind);
    ek[o][t] = (int8_t)kind; ep[o][t] = pid;
    knownEdges++; if (kind == K_KELP) knownKelp++;
    if (kind != K_OPEN) dirtyEdge(o, t);   // unknown was already planned as open
    if (symKnown) {
      int m = mirrorE(o, t, symKnown);
      if (ek[o][m] == K_UNK) { ek[o][m] = (int8_t)kind; ep[o][m] = -1; if (kind != K_OPEN) dirtyEdge(o, m); }
    }
  } else if (direct && (cur != kind || (kind == K_PORT && ep[o][t] != pid))) {
    // our eyes beat mirrors and sonar
    if (cur != kind && kind != K_PORT) { epart[o][t] = -1; }
    ek[o][t] = (int8_t)kind; ep[o][t] = pid;
    dirtyEdge(o, t);
  }
  if (kind == K_PORT && pid >= 0) registerPortal(pid, o, t);
}

// ============================================================================
//  OBSERVATION -> MEMORY
// ============================================================================
void Bot::integrate() {
  turnStamp++;
  turnsAlive++;
  headT = tileOf(wx[24], wy[24]);
  if (firstRound < 0) {
    firstRound = round;
    isAnchor = (round <= 1);  // only initial dragons exist in round 1 (we never split then)
    startTile = headT;
    salt = mix32(0x5EAD5EADu + (u32)team * 0x1000193u);
  }
  // ---- tiles ----
  int occN = 0;
  for (int i = 0; i < 49; i++) {
    int t = tileOf(wx[i] % W, wy[i] % H);
    winStamp[t] = turnStamp;
    int spawn = wc[i] >= 0 ? 1 : 0;
    bool first = seenR[t] < 0;
    if (first) { spawnKnown++; if (spawn) spawnYes++; }
    // countdown bookkeeping: only "new information" feeds symmetry evidence
    bool newInfo = first;
    if (!first && spawn) {
      int expect = cdV[t] - (round - cdR[t]);
      if (cdR[t] < 0 || expect != wc[i]) {
        newInfo = true;
        if (cdR[t] >= 0 && wc[i] > expect) { gapSum += wc[i]; gapCnt++; }  // fresh draw
      }
    }
    if (newInfo) symEvidenceTile(t, spawn, wc[i]);
    seenR[t] = (int16_t)round;
    spawnK[t] = (int8_t)spawn;
    if (spawn) { cdR[t] = (int16_t)round; cdV[t] = (int16_t)wc[i]; }
    if (wp[i]) pearlR[t] = (int16_t)round; else emptyR[t] = (int16_t)round;
    futureCd[t] = (!wp[i] && spawn) ? (int8_t)std::min(wc[i], 100) : (int8_t)-1;
    if (symKnown) mirrorTileInfo(t);
  }
  // ---- edges ----
  for (int r = 0; r < 8; r++) {
    int y = (r < 7) ? wy[r * 7] : (wy[42] + 1) % H;
    for (int c = 0; c < 7; c++) {
      if (hK[r][c] == K_UNK) continue;
      recordEdge(0, tileOf(wx[c], y), hK[r][c], hP[r][c], true);
    }
  }
  for (int r = 0; r < 7; r++) {
    int y = wy[r * 7];
    for (int c = 0; c < 8; c++) {
      if (vK[r][c] == K_UNK) continue;
      int x = (c < 7) ? wx[c] : (wx[6] + 1) % W;
      recordEdge(1, tileOf(x, y), vK[r][c], vP[r][c], true);
    }
  }
  decideSymmetry();
  for (auto& s : rsegs) {
    int t = tileOf(s.x, s.y);
    if (t >= 0 && t < NT) { occR[t] = (int16_t)round; if (s.id != myId) occN++; }
  }
  density = occN / 49.0;
  kelpFrac = knownEdges > 0 ? (double)knownKelp / knownEdges : 0.0;
}

// Follow segment facings from the head to recover a dragon's body order.
static void buildChainFor(const Bot& b, const std::vector<const Bot::RawSeg*>& sg, std::vector<int>& chain,
                          int& headIdx) {
  chain.clear();
  headIdx = -1;
  for (size_t i = 0; i < sg.size(); i++) if (sg[i]->head) headIdx = (int)i;
  if (headIdx < 0) return;
  std::vector<char> used(sg.size(), 0);
  std::vector<int> succ(sg.size(), -3), tiles(sg.size());
  for (size_t i = 0; i < sg.size(); i++) {
    tiles[i] = b.tileOf(sg[i]->x, sg[i]->y);
    if (!sg[i]->head) succ[i] = b.stepMem(tiles[i], sg[i]->face);
  }
  int cur = headIdx;
  used[cur] = 1;
  chain.push_back(tiles[cur]);
  for (;;) {
    int nxt = -1;
    for (size_t i = 0; i < sg.size(); i++)
      if (!used[i] && succ[i] == tiles[cur]) { nxt = (int)i; break; }
    if (nxt < 0) break;
    used[nxt] = 1;
    chain.push_back(tiles[nxt]);
    cur = nxt;
  }
}

void Bot::rebuildTrailFromWindow() {
  std::vector<const RawSeg*> mine;
  for (auto& s : rsegs) if (s.id == myId) mine.push_back(&s);
  int hi;
  buildChainFor(*this, mine, trail, hi);
  if (trail.empty()) trail.push_back(headT);
  if ((int)trail.size() > length) trail.resize(length);
  bodyKnown = (int)trail.size();
}

void Bot::updateTrail() {
  if (lastKind < 0 || trail.empty()) { rebuildTrailFromWindow(); return; }
  if (lastKind == 0) {
    if (headT == expectHead && !lastPath.empty()) {
      for (int t : lastPath) trail.insert(trail.begin(), t);
    } else {
      trail.insert(trail.begin(), headT);  // e.g. we went through a portal we could not predict
    }
  }
  if ((int)trail.size() > length) trail.resize(length);
  bodyKnown = (int)trail.size();
  // verify against what we can see; rebuild on any disagreement
  bool bad = trail.empty() || trail[0] != headT;
  std::vector<int> ownVis;
  for (auto& s : rsegs) if (s.id == myId) ownVis.push_back(tileOf(s.x, s.y));
  if (!bad) {
    for (int i = 0; i < bodyKnown && !bad; i++) {
      int t = trail[i];
      if (!inWin(t)) continue;
      bool found = false;
      for (int v : ownVis) if (v == t) { found = true; break; }
      if (!found) bad = true;
    }
    if (!bad && bodyKnown == length) {
      for (int v : ownVis) {
        bool found = false;
        for (int i = 0; i < bodyKnown; i++) if (trail[i] == v) { found = true; break; }
        if (!found) { bad = true; break; }
      }
    }
  }
  if (bad) rebuildTrailFromWindow();
}

// ============================================================================
//  PER-TURN WORLD MODEL
// ============================================================================
void Bot::buildNeighbors() {
  if (nbrAll) {
    for (int t = 0; t < NT; t++)
      for (int d = 0; d < 4; d++) nbr[t * 4 + d] = stepMem(t, d);
    nbrAll = false;
  } else {
    for (int t : nbrDirty)
      for (int d = 0; d < 4; d++) nbr[t * 4 + d] = stepMem(t, d);
  }
  nbrDirty.clear();
}

void Bot::buildOccupancy() {
  std::fill(occ.begin(), occ.end(), -1);
  std::fill(occTeam.begin(), occTeam.end(), (int8_t)-1);
  std::fill(occHead.begin(), occHead.end(), (int8_t)0);
  std::fill(occOwn.begin(), occOwn.end(), (int8_t)0);
  std::fill(blockT.begin(), blockT.end(), 0);
  // own body: segment i frees at turn L-i+1
  for (int i = 0; i < bodyKnown; i++) {
    int t = trail[i];
    occ[t] = myId; occTeam[t] = (int8_t)team; occOwn[t] = 1;
    blockT[t] = length - i + 1;
  }
  occHead[headT] = 1;
  // remembered rear segments outside the window
  if (bodyKnown >= length) {
    for (int t : ownMemList) ownMemUntil[t] = -1;
    ownMemList.clear();
  } else {
    size_t w = 0;
    for (size_t i = 0; i < ownMemList.size(); i++) {
      int t = ownMemList[i];
      if (ownMemUntil[t] <= round || inWin(t)) { ownMemUntil[t] = -1; continue; }
      ownMemList[w++] = t;
      if (!occOwn[t]) {
        occ[t] = myId; occTeam[t] = (int8_t)team; occOwn[t] = 1;
        blockT[t] = std::max(2, ownMemUntil[t] - round + 1);
      }
    }
    ownMemList.resize(w);
  }
  // group other segments by dragon
  others.clear();
  std::vector<int> ids;
  for (auto& s : rsegs) if (std::find(ids.begin(), ids.end(), s.id) == ids.end()) ids.push_back(s.id);
  for (int id : ids) {
    std::vector<const RawSeg*> sg;
    for (auto& s : rsegs) if (s.id == id) sg.push_back(&s);
    if (id == myId) {
      // own segments not in the known trail (unknown part of our tail)
      for (auto* s : sg) {
        int t = tileOf(s->x, s->y);
        if (!occOwn[t]) {
          occ[t] = myId; occTeam[t] = (int8_t)team; occOwn[t] = 1;
          blockT[t] = std::max(2, length - bodyKnown + 1);
          if (ownMemUntil[t] < 0) ownMemList.push_back(t);
          ownMemUntil[t] = (int16_t)(round + (length - bodyKnown));
        }
      }
      continue;
    }
    Other o;
    o.id = id; o.team = sg[0]->team; o.enemy = (o.team != team);
    o.visible = (int)sg.size();
    int hi;
    buildChainFor(*this, sg, o.chain, hi);
    if (hi >= 0) { o.head = tileOf(sg[hi]->x, sg[hi]->y); o.face = sg[hi]->face; }
    o.lenLow = o.visible;
    if (!o.chain.empty() && (int)o.chain.size() == o.visible) {
      // is the last chain element provably the tail?
      int T = o.chain.back();
      bool ok = true;
      for (int d = 0; d < 4 && ok; d++) {
        int ot, et; edgeOf(T, d, ot, et);
        if (ek[ot][et] == K_PORT) ok = false;
        int nb = nbPlain(T, d);
        if (!inWin(nb)) ok = false;
      }
      if (ok) { o.tailSeen = true; o.lenKnown = o.visible; }
    }
    for (auto* s : sg) {
      int t = tileOf(s->x, s->y);
      occ[t] = id; occTeam[t] = (int8_t)o.team; occHead[t] = (int8_t)(s->head ? 1 : 0);
      blockT[t] = INF;
    }
    if (o.lenKnown > 0)
      for (int i = 0; i < (int)o.chain.size(); i++) blockT[o.chain[i]] = o.lenKnown - i + 1;
    else if (!o.chain.empty()) {
      // unknown length: segments near the head stay at least lenLow-i+1 turns
      for (int i = 0; i < (int)o.chain.size(); i++) blockT[o.chain[i]] = INF;
    }
    others.push_back(o);
  }
  // Learn how aggressive the opponent is.
  //  * an enemy head that jumped >= 2 tiles sprinted (kamikaze bots only
  //    sprint to strike) -> evidence of aggression;
  //  * since our last turn every enemy moved exactly once while our head sat
  //    on headT: if it could see headT and reach it with a paid sprint but
  //    did not, it passed on a strike -> evidence of passivity.
  std::vector<PrevEnemy> now;
  for (auto& o : others) {
    if (!o.enemy || o.head < 0) continue;
    now.push_back({o.id, o.head, o.lenKnown > 0 ? o.lenKnown : o.lenLow});
    for (auto& p : prevEnemies) {
      if (p.id != o.id) continue;
      if (manhattan(p.head, o.head) >= 2 && inWin(p.head)) {
        bool portalNear = false;
        for (int d = 0; d < 4; d++) { int ot, et; edgeOf(o.head, d, ot, et); if (ek[ot][et] == K_PORT) portalNear = true; }
        if (!portalNear) aggrHit += 1.0;
      }
      if (turnsAlive > 1 && chebyshev(p.head, headT) <= 3 && p.len >= 2) {
        int md = manhattan(p.head, headT);
        if (md <= p.len - 1 && md <= 4) {
          // verify a free path of that length existed (static check)
          std::fill(dist2.begin(), dist2.end(), INF);
          int h = 0, tl = 0;
          dist2[p.head] = 0; bfsQ[tl++] = p.head;
          bool reach = false;
          while (h < tl && !reach) {
            int u = bfsQ[h++];
            if (dist2[u] >= p.len - 1) continue;
            for (int k = 0; k < 4; k++) {
              int v = nbr[u * 4 + k];
              if (v < 0 || dist2[v] != INF) continue;
              dist2[v] = dist2[u] + 1;
              if (v == headT) { reach = true; break; }
              if (occ[v] >= 0 && v != p.head) continue;
              bfsQ[tl++] = v;
            }
          }
          if (reach) aggrPass += 1.0;
        }
      }
    }
  }
  prevEnemies = now;
  // Witnessed kamikaze: a friendly head and an enemy head that could reach it
  // were both visible last turn and both have vanished from tiles we can still
  // see (a head-to-head kills both; nothing else removes two dragons at once).
  auto visibleNow = [&](int id) {
    for (auto& o : others) if (o.id == id) return true;
    return false;
  };
  if (turnsAlive > 1) {
    for (auto& f : lastSeen) {
      if (f.team != team || f.id == myId || !inWin(f.head) || visibleNow(f.id)) continue;
      for (auto& e : lastSeen) {
        if (e.team == team || !inWin(e.head) || visibleNow(e.id)) continue;
        // our dragons only strike enemies longer than themselves, so if the
        // vanished friend was the longer one the enemy started it
        if (manhattan(e.head, f.head) <= std::max(1, e.len - 1) && chebyshev(e.head, f.head) <= 3 && f.len > e.len) {
          aggrHit += 2.0;
          break;
        }
      }
    }
  }
  lastSeen.clear();
  for (auto& o : others)
    if (o.head >= 0) lastSeen.push_back({o.id, o.team, o.head, o.lenKnown > 0 ? o.lenKnown : o.lenLow});
}

void Bot::buildDistances() {
  // multi-source BFS from other heads (competition / Voronoi), and enemies only
  for (int pass = 0; pass < 2; pass++) {
    std::vector<int>& D = pass == 0 ? dOther : dEnemy;
    std::fill(D.begin(), D.end(), INF);
    int h = 0, tl = 0;
    for (auto& o : others) {
      if (o.head < 0) continue;
      if (pass == 1 && !o.enemy) continue;
      D[o.head] = 0; bfsQ[tl++] = o.head;
    }
    while (h < tl) {
      int u = bfsQ[h++];
      for (int k = 0; k < 4; k++) {
        int v = nbr[u * 4 + k];
        if (v < 0 || D[v] != INF) continue;
        if (occ[v] >= 0) continue;
        D[v] = D[u] + 1; bfsQ[tl++] = v;
      }
    }
  }
  // threat: expected loss (in length units) if our head ends a turn on a tile
  // The ladder meta uses kamikaze sprints, so any enemy head that can see a
  // tile and reach it this turn is assumed willing to trade into it.
  std::fill(threat.begin(), threat.end(), 0.f);
  threatHere = 0.f; threatTail = 0.f;
  int tailT = bodyKnown == length && bodyKnown > 0 ? trail[bodyKnown - 1] : -1;
  double ag = aggression();
  double pAdj = std::min(0.95, ag * P.pAdj / std::max(0.05, P.aggrPrior)), pFar = ag * P.pFar / std::max(0.05, P.aggrPrior);
  pFar = std::min(pFar, 0.95);
  // Sources of threat: visible enemy heads, plus "hidden heads" just outside
  // the window where a visible enemy body leads (its head is beyond).
  struct Src { int t; int reach; int lenLow; double scale; };
  std::vector<Src> srcs;
  // Ghosts: enemy heads we saw a few rounds ago that are out of sight now.
  // Each round away lets them come one tile closer, so the reach grows.
  {
    std::vector<Ghost> keep;
    for (auto& g : ghosts) {
      bool visible = false;
      for (auto& o : others) if (o.id == g.id && o.head >= 0) visible = true;
      int age = round - g.r;
      if (visible || age > P.ghostAge) continue;
      keep.push_back(g);
      if (unitCount <= P.ghostUnits && kamikazeEvidence())
        srcs.push_back({g.head, std::min(g.reach + age, 8), g.len, std::pow(P.ghostDecay, age)});
    }
    ghosts = keep;
    for (auto& o : others)
      if (o.enemy && o.head >= 0)
        ghosts.push_back({o.id, o.head, o.lenLow, o.lenKnown > 0 ? o.lenKnown - 1 : 6, round});
  }
  for (auto& o : others) {
    if (!o.enemy) continue;
    if (o.head >= 0) {
      int reach = o.lenKnown > 0 ? o.lenKnown - 1 : 6;
      srcs.push_back({o.head, std::min(reach, 8), o.lenLow, 1.0});
      continue;
    }
  }
  for (auto& src : srcs) {
    int reach = src.reach;
    if (reach < 1) continue;
    // net loss of a 1-for-1 trade, never below a small floor
    double loss = src.scale * std::max(0.15 * myWorth(), myWorth() - P.strikeGain * enemyWorth(src.lenLow));
    Other ofake; ofake.head = src.t;
    Other& o = ofake;
    // depth-limited BFS from the enemy head through free tiles (and our head)
    std::fill(dist2.begin(), dist2.end(), INF);
    int h = 0, tl = 0;
    dist2[o.head] = 0; bfsQ[tl++] = o.head;
    while (h < tl) {
      int u = bfsQ[h++];
      if (dist2[u] >= reach) continue;
      for (int k = 0; k < 4; k++) {
        int v = nbr[u * 4 + k];
        if (v < 0 || dist2[v] != INF) continue;
        dist2[v] = dist2[u] + 1;
        if (occ[v] >= 0 && !(occOwn[v] && blockT[v] <= 2)) continue;  // cannot pass bodies (our tail moves)
        bfsQ[tl++] = v;
      }
    }
    if (dist2[headT] <= reach && chebyshev(o.head, headT) <= 3)
      threatHere += (float)((dist2[headT] == 1 ? pAdj : pFar) * loss);
    if (tailT >= 0 && dist2[tailT] <= reach && chebyshev(o.head, tailT) <= 3)
      threatTail += (float)((dist2[tailT] == 1 ? pAdj : pFar) * loss);
    for (int i = 0; i < tl; i++) {
      int t = bfsQ[i];
      int d = dist2[t];
      if (d == 0 || d > reach) continue;
      if (src.scale >= 1.0 && chebyshev(o.head, t) > 3) continue;  // it cannot see us there
      // longer sprints are costlier for the striker and easier to spoil
      double p = d == 1 ? pAdj : pFar * std::max(0.3, 1.0 - P.pDecay * (d - 2));
      threat[t] += (float)(p * loss);
    }
    // tiles next to its head (after our move our head could be there)
    for (int k = 0; k < 4; k++) {
      int v = nbr[o.head * 4 + k];
      if (v >= 0 && occ[v] < 0 && dist2[v] == INF) threat[v] += (float)(pAdj * loss);
    }
  }
}

// How likely is a pearl on tile t by the time we could get there?
void Bot::buildPearlValues() {
  double prior = 0.22;
  if (spawnKnown > 20) prior = 0.35 * (double)spawnYes / spawnKnown;
  for (int t = 0; t < NT; t++) {
    float v = 0.f;
    if (inWin(t)) {
      if (occ[t] >= 0) v = 0.f;
      else if (pearlR[t] == round) v = 1.0f;
      else v = 0.f;  // future spawns handled with futureCd in the field
    } else if (seenR[t] >= 0) {
      int age = round - seenR[t];
      if (spawnK[t] == 0) v = 0.f;
      else if (pearlR[t] == seenR[t]) v = (float)(0.8 * std::exp(-age / 50.0));
      else {
        int pred = cdR[t] >= 0 ? cdR[t] + cdV[t] : seenR[t] + 12;
        if (round >= pred) v = (float)(0.6 * std::exp(-(round - pred) / 90.0) + 0.1);
        else v = 0.08f;
      }
      if (occR[t] >= round - 2 && occR[t] == seenR[t]) v *= 0.3f;  // was under a dragon
    } else {
      int m = symKnown ? mirrorT(t, symKnown) : -1;
      if (m >= 0 && seenR[m] >= 0) {
        if (spawnK[m] == 0) v = 0.f;
        else if (cdShared && cdR[m] >= 0) {
          int pred = cdR[m] + cdV[m];
          v = round >= pred ? 0.6f : 0.15f;
        } else v = (float)(prior * 1.4);
      } else v = (float)prior;
    }
    pval[t] = v;
  }
  // Hunting: enemy heads we would gladly trade for pull us to within our
  // sprint reach (tiles next to their head), sonar sightings pull from afar.
  if (!anchorNow() && length >= 3 && length < P.growLen) {
    for (auto& o : others) {
      if (!o.enemy || o.head < 0) continue;
      if (P.strikeGain * enemyWorth(o.lenLow) < myWorth() + 1.0) continue;
      for (int k = 0; k < 4; k++) {
        int v = nbr[o.head * 4 + k];
        if (v >= 0 && occ[v] < 0) pval[v] += (float)P.huntW;
      }
    }
    for (auto& s : sightings)
      if (round - s.r <= 3 && s.t >= 0 && s.t < NT && P.strikeGain * enemyWorth(s.len) >= myWorth() + 1.0)
        pval[s.t] += (float)P.huntW;
  }
}

// ============================================================================
//  SEARCH PRIMITIVES
// ============================================================================
// Time-aware flood fill from `src` where our head sits after this turn.
// A tile v may be entered on turn s (s = 2 is our next turn) iff blockT[v] <= s
// (own segments shifted by ownShift). Returns the number of reachable tiles,
// fills d[] with move counts and counts tiles we reach strictly before any
// other visible head (territory).
int Bot::timedBfs(int src, int s0, int ownShift, std::vector<int>& d, int& territory, int cap) {
  std::fill(d.begin(), d.end(), INF);
  int h = 0, tl = 0, cnt = 0;
  territory = 0;
  d[src] = 0;
  bfsQ[tl++] = src;
  // Pearls we may be forced to eat on the way stop our tail for a turn each,
  // so our own segments free up later.  Count pearls met at smaller
  // distances (an upper bound on the pearls on any shortest path).  This is
  // what makes pockets with pearls (or pearls about to spawn) look as tight
  // as they really are.
  int pearlsBefore = 0, pearlsLayer = 0, layer = 0;
  while (h < tl) {
    int u = bfsQ[h++];
    cnt++;
    if (d[u] < dOther[u]) territory++;
    int du = d[u];
    if (du != layer) { pearlsBefore += pearlsLayer; pearlsLayer = 0; layer = du; }
    bool pu = du > 0 && occ[u] < 0 &&
              ((pearlR[u] == seenR[u] && seenR[u] >= 0 && (inWin(u) || round - seenR[u] < 30)) ||
               (inWin(u) && futureCd[u] > 0 && futureCd[u] <= du + 1));
    if (pu) pearlsLayer++;
    // Past the pearl horizon only keep going while we still need to prove
    // that there is enough room for our body.
    if (du >= 32 && cnt >= cap) continue;
    int enterAt = s0 + du + 1;
    int pdelay = std::min(P.pearlDelayCap, pearlsBefore + (pu ? 1 : 0));
    for (int k = 0; k < 4; k++) {
      int v = nbr[u * 4 + k];
      if (v < 0 || d[v] != INF) continue;
      int b = blockT[v];
      if (b) {
        if (occOwn[v]) b += ownShift + pdelay;
        if (b > enterAt) continue;
      }
      d[v] = du + 1;
      bfsQ[tl++] = v;
    }
  }
  return cnt;
}

// Like timedBfs but never expands through a tile another head reaches first
// (or at the same time): the space we own no matter what the others do.
int Bot::voronoiBfs(int src, int ownShift, int enough) {
  std::vector<int>& d = dist3;
  std::fill(d.begin(), d.end(), INF);
  int h = 0, tl = 0;
  d[src] = 0;
  bfsQ[tl++] = src;
  while (h < tl && tl < enough) {
    int u = bfsQ[h++];
    int du = d[u];
    for (int k = 0; k < 4; k++) {
      int v = nbr[u * 4 + k];
      if (v < 0 || d[v] != INF) continue;
      int b = blockT[v];
      if (b) {
        if (occOwn[v]) b += ownShift;
        if (b > du + 2) continue;
      }
      if (dOther[v] <= du + 1) continue;  // someone else gets there first
      d[v] = du + 1;
      bfsQ[tl++] = v;
    }
  }
  return tl;
}

// Cost of boxing in a visible teammate: if our head on `dest` leaves a
// teammate less room than its body needs, we pay in proportion to its size.
double Bot::courtesyCost(int dest) {
  double cost = 0;
  for (auto& o : others) {
    if (o.head < 0) continue;
    if (o.enemy) continue;
    int need = std::max(3, (o.lenKnown > 0 ? o.lenKnown : o.lenLow) + 1);
    int saved = blockT[dest];
    int savedOcc = occ[dest];
    blockT[dest] = INF; occ[dest] = myId;
    // static flood fill from the teammate's head, capped at `need`
    std::vector<int>& d = dist2;
    std::fill(d.begin(), d.end(), INF);
    int h = 0, tl = 0, cnt = 0;
    d[o.head] = 0; bfsQ[tl++] = o.head;
    while (h < tl && cnt < need) {
      int u = bfsQ[h++];
      if (u != o.head) cnt++;
      for (int k = 0; k < 4; k++) {
        int v = nbr[u * 4 + k];
        if (v < 0 || d[v] != INF) continue;
        int b = blockT[v];
        if (b && b > d[u] + 2) continue;
        d[v] = d[u] + 1;
        bfsQ[tl++] = v;
      }
    }
    blockT[dest] = saved; occ[dest] = savedOcc;
    if (cnt < need) cost += 12.0 + 1.5 * (need - cnt) + 0.5 * need;
  }
  return cost;
}

// Exact replay of the engine's step rules for our own dragon, using only what
// is visible this turn. Returns false if the action would kill us (except a
// deliberate head-to-head into an enemy head on the last step).
bool Bot::simulate(const std::string& steps, Cand& c) {
  c.path.clear();
  c.ate = false; c.pearlsOnPath = 0; c.attack = false; c.unknownExit = false;
  std::vector<int> body(trail.begin(), trail.begin() + bodyKnown);
  int unknownTail = length - bodyKnown;
  int len = length;
  int cur = headT;
  int k = (int)steps.size();
  for (int j = 0; j < k; j++) {
    if (j > 0 && len <= 2) return false;  // cannot pay for another step
    int d = steps[j] == 'N' ? DIR_N : steps[j] == 'E' ? DIR_E : steps[j] == 'S' ? DIR_S : DIR_W;
    if (!inWin(cur)) return false;
    int nx = nbr[cur * 4 + d];
    if (nx == -1) return false;           // kelp
    if (nx == -2) {                       // portal, exit unknown
      if (k != 1) return false;
      c.unknownExit = true; c.dest = -2; c.path.push_back(-2);
      return true;
    }
    if (!inWin(nx)) {                     // portal exit we cannot see
      if (k != 1) return false;
      c.unknownExit = true; c.dest = nx; c.path.push_back(nx);
      return true;
    }
    // own body first (tail included: it has not moved yet)
    for (int i = 0; i < (int)body.size(); i++) if (body[i] == nx) return false;
    if (occOwn[nx]) {
      bool inBody = false;
      for (int b : body) if (b == nx) inBody = true;
      if (!inBody && unknownTail > 0) return false;
    }
    if (occ[nx] >= 0 && occ[nx] != myId) {
      if (occHead[nx] && occTeam[nx] != team && j == k - 1) {
        c.attack = true; c.attackId = occ[nx]; c.path.push_back(nx); c.dest = nx;
        return true;
      }
      return false;
    }
    bool pearl = (pearlR[nx] == round);
    body.insert(body.begin(), nx);
    if (pearl) { c.pearlsOnPath++; if (j == 0) c.ate = true; }
    auto popTail = [&]() {
      if (unknownTail > 0) unknownTail--;
      else if (!body.empty()) body.pop_back();
    };
    if (!pearl) popTail();
    else len++;
    if (j > 0) { popTail(); len--; }
    c.path.push_back(nx);
    cur = nx;
  }
  c.dest = cur;
  return true;
}

// Shortest in-window sprint to an enemy head (our own tail retreats 2 tiles
// per extra step while sprinting).
static bool findSprintPath(Bot& b, int target, std::string& out) {
  int NT = b.NT;
  std::vector<int>& d = b.dist2;
  std::fill(d.begin(), d.end(), INF);
  std::vector<int> par(NT, -1), pdir(NT, -1);
  std::vector<int> ownIdx(NT, -1);
  for (int i = 0; i < b.bodyKnown; i++) ownIdx[b.trail[i]] = i;
  int h = 0, tl = 0;
  d[b.headT] = 0; b.bfsQ[tl++] = b.headT;
  while (h < tl) {
    int u = b.bfsQ[h++];
    if (u == target) break;
    int step = d[u] + 1;
    if (step > b.length - 1 + 2) continue;
    for (int k = 0; k < 4; k++) {
      int v = b.nbr[u * 4 + k];
      if (v < 0 || d[v] != INF || !b.inWin(v)) continue;
      if (v == target) { d[v] = step; par[v] = u; pdir[v] = k; b.bfsQ[tl++] = v; continue; }
      if (b.occ[v] >= 0) {
        if (!b.occOwn[v] || ownIdx[v] < 0) continue;
        int i = ownIdx[v];
        if (!(step >= 2 && i > b.length - 2 * (step - 1))) continue;
      }
      d[v] = step; par[v] = u; pdir[v] = k; b.bfsQ[tl++] = v;
    }
  }
  if (d[target] == INF) return false;
  out.clear();
  for (int v = target; v != b.headT; v = par[v]) out += DCH[pdir[v]];
  std::reverse(out.begin(), out.end());
  return true;
}

// ============================================================================
//  ARENA-ADAPTIVE POLICY
// ============================================================================
// How many dragons the team should field on this arena. Small arenas saturate
// quickly (every body is a wall), big open arenas reward a large swarm.
int Bot::unitCap() const {
  double area = (double)W * H;
  // A fresh process knows little about the arena: be conservative until it
  // has seen a fair sample of tiles (kelp share and pearl richness).
  double open = 1.0 - std::min(0.5, kelpFrac * 2.5);
  if (knownEdges < 300) open = std::min(open, 0.75);
  double rich = 0.8;
  if (spawnKnown > 60) {
    double frac = (double)spawnYes / spawnKnown;
    double gap = gapCnt > 3 ? gapSum / gapCnt : 14.0;
    double rate = frac / std::max(3.0, gap);        // pearls per tile per round
    rich = std::clamp(rate / 0.05, 0.5, 1.6);
  }
  double cap = area / P.capDiv * open * rich;
  if (area <= 170) cap = std::min(cap, (double)P.capSmallMax);
  cap = std::clamp(cap, (double)P.capMin, (double)std::min(unitLimit, 64));
  return (int)cap;
}

int Bot::splitLen() const {
  double area = (double)W * H;
  if (area <= 300) return P.splitLenSmall;
  return P.splitLen;
}

// Would SPLIT n leave both halves with room to live?
bool Bot::splitSafe(int n, double& quality) {
  quality = 0;
  if (bodyKnown != length || n < 2 || length - n < 2) return false;
  int L = length;
  // temporarily re-time our own tiles for the two new dragons
  std::vector<int> saved(L);
  for (int i = 0; i < L; i++) saved[i] = blockT[trail[i]];
  for (int i = 0; i < L - n; i++) blockT[trail[i]] = (L - n) - i + 1;
  for (int i = L - n; i < L; i++) { int j = L - 1 - i; blockT[trail[i]] = n - j + 1; }
  int childHead = trail[L - 1];
  // the parent does not move this turn and the child appears on our tail:
  // neither may be sitting inside an enemy's strike range
  if (threatHere > 0.f || threatTail > 0.f) return false;
  double best[2] = {0, 0};
  for (int who = 0; who < 2; who++) {
    int from = who == 0 ? headT : childHead;
    int need = (who == 0 ? L - n : n) + 2;
    for (int k = 0; k < 4; k++) {
      int v = nbr[from * 4 + k];
      if (v < 0 || occ[v] >= 0) continue;
      if (threat[v] > 0.5f * (who == 0 ? L - n : n)) continue;
      int terr;
      int sp = timedBfs(v, 1, 0, dist2, terr, 2 * need + 10);
      double q = (double)sp / need;
      best[who] = std::max(best[who], std::min(2.0, q));
    }
  }
  for (int i = 0; i < L; i++) blockT[trail[i]] = saved[i];
  quality = std::min(best[0], best[1]);
  return quality >= 1.0;
}

// Room (timed flood fill from its best exit) for a child made of our rear n
// segments.  The child's head is our tail; it moves later this round.
int Bot::childRoom(int n, int cap) {
  if (bodyKnown != length || n < 2 || length - n < 2) return 0;
  int L = length;
  std::vector<int> saved(L);
  for (int i = 0; i < L; i++) saved[i] = blockT[trail[i]];
  for (int i = 0; i < L - n; i++) blockT[trail[i]] = (L - n) - i + 1;
  for (int i = L - n; i < L; i++) { int j = L - 1 - i; blockT[trail[i]] = n - j + 1; }
  int ch = trail[L - 1];
  int bestSp = 0;
  for (int k = 0; k < 4; k++) {
    int v = nbr[ch * 4 + k];
    if (v < 0 || occ[v] >= 0) continue;
    if (dOther[v] <= 1) continue;  // another head can take this exit before the child moves
    int terr;
    bestSp = std::max(bestSp, timedBfs(v, 1, 0, dist2, terr, cap));
  }
  for (int i = 0; i < L; i++) blockT[trail[i]] = saved[i];
  return bestSp;
}

// ============================================================================
//  SCORING
// ============================================================================
#define W_FIELD P.wField
#define W_EAT P.wEat
#define W_TRAP P.wTrap
#define W_SPACE P.wSpace
#define W_TERR P.wTerr
#define W_RISK P.wRisk
#define W_LEN P.wLen
#define W_CONTEST P.wContest

double Bot::evaluate(Cand& c) {
  double importance = anchorNow() ? 2.2 : (length >= 24 ? 1.8 : length >= 12 ? 1.3 : 1.0);
  int k = (int)c.steps.size();
  int newLen = length + (c.ate ? 1 : 0);
  for (int j = 1; j < k; j++) newLen -= 1;
  newLen += std::max(0, c.pearlsOnPath - (c.ate ? 1 : 0));
  double lost = (double)(length - newLen) + (c.ate ? 1 : 0);
  if (c.unknownExit) {
    // Blind portal crossing: we cannot see the exit tile.
    double pOcc = std::min(0.9, 0.04 + density * 1.3);
    if (c.dest >= 0 && seenR[c.dest] >= round - 2 && occR[c.dest] < seenR[c.dest]) pOcc *= 0.4;
    double sc = 1.0 - pOcc * (60.0 + 4.0 * length) * importance;
    if (c.dest == -2) sc += 4.0 * (1.0 - density);  // learning where it leads
    return sc;
  }
  // re-time the tiles we walk over in a sprint
  std::vector<int> saved;
  for (int j = 1; j < k; j++) {
    int t = c.path[j - 1];
    saved.push_back(blockT[t]);
    blockT[t] = newLen - k + j + 2;  // absolute turn at which this new segment frees
  }
  int shift = newLen - k - length + 1;
  int terr;
  int space = timedBfs(c.dest, 1, shift, dist, terr, 4 * newLen + 40);
  // Food field over every reachable tile: the best pearl dominates (so we
  // always converge on eating it) and the rest add a mild pull to clusters.
  // The pearl under the new head counts at distance 0 (eaten now).
  double best = 0, sum = 0;
  for (int i = 0; i < space; i++) {
    int t = bfsQ[i];
    int dm = dist[t];
    double v = pval[t];
    if (dm == 0) v = c.ate ? 1.0 : 0.0;
    if (inWin(t) && futureCd[t] > 0 && occ[t] < 0) {
      double fv = dm >= futureCd[t] ? 0.85 : 0.2;
      if (fv > v) v = fv;
    }
    if (v <= 0) continue;
    int dO = dOther[t];
    if (dm > 0) {
      if (dO < dm) v *= 0.3;
      else if (dO == dm) v *= 0.6;
    }
    double cv = v * gpow[std::min(dm, NT)];
    sum += cv;
    if (cv > best) best = cv;
  }
  double field = best + 0.35 * (sum - best);
  for (int j = 1; j < k; j++) blockT[c.path[j - 1]] = saved[j - 1];
  int vor = voronoiBfs(c.dest, shift, newLen + 2);
  double court = courtesyCost(c.dest);
  c.space = space; c.terr = terr; c.field = field; c.vor = vor; c.courtesy = court;
  double sc = W_FIELD * field;
  if (c.ate) sc += W_EAT;
  int need = newLen + 1;
  // Our own segments we have never seen (a child's rear half outside the
  // window) are somewhere in that "free" room: do not count on it.
  int effSpace = space - std::max(0, length - bodyKnown);
  // Risks below are priced in what this dragon is worth, so a long dragon
  // (the likely tiebreak winner) will not trade safety for a pearl while a
  // two-segment dragon still forages boldly.
  double worth = myWorth();
  if (effSpace < need) sc -= W_TRAP * (1.0 + (double)(need - effSpace) / need) + worth;
  else {
    sc += W_SPACE * std::log(1.0 + std::min(effSpace, 4 * newLen + 24));
    // room we own outright; short of our length means others can seal us in
    if (vor < need) {
      double sh = (double)(need - vor) / need;
      sc -= W_CONTEST * sh;
    }
  }
  sc -= court;
  sc += W_TERR * std::min(terr, 200) / 200.0;
  sc -= threat[c.dest] * W_RISK;
  sc -= W_LEN * lost * (newLen <= 3 ? P.tinySprint : 1.0);  // tiny dragons also lose the ability to split
  if (lastTarget >= 0 && dist[lastTarget] < INF && pval[lastTarget] > 0.5f) sc += 0.3;
  c.score = sc;
  return sc;
}

void Bot::genCandidates(std::vector<Cand>& out) {
  for (int d = 0; d < 4; d++) {
    Cand c;
    c.steps = std::string(1, DCH[d]);
    c.tag = "step";
    if (simulate(c.steps, c)) out.push_back(c);
  }
  for (auto& o : others) {
    if (!o.enemy || o.head < 0) continue;
    std::string p;
    if (!findSprintPath(*this, o.head, p)) continue;
    if (p.size() < 1 || (int)p.size() > length + 2) continue;
    Cand c;
    c.steps = p;
    if (simulate(p, c) && c.attack && c.attackId == o.id) { c.tag = "strike"; out.push_back(c); }
  }
}

// Depth-limited enumeration of multi-step escapes (only when trapped).
static void genSprints(Bot& b, std::vector<Cand>& out, int maxSteps) {
  std::string s;
  std::vector<std::string> seqs;
  for (int L = 2; L <= maxSteps; L++) {
    int total = 1;
    for (int i = 0; i < L; i++) total *= 4;
    for (int code = 0; code < total; code++) {
      s.clear();
      int c = code;
      bool ok = true;
      for (int i = 0; i < L; i++) {
        int d = c & 3; c >>= 2;
        if (i > 0 && d == (s.back() == 'N' ? DIR_S : s.back() == 'S' ? DIR_N : s.back() == 'E' ? DIR_W : DIR_E)) ok = false;
        s += DCH[d];
      }
      if (ok) seqs.push_back(s);
    }
  }
  for (auto& q : seqs) {
    Cand c;
    c.steps = q;
    c.tag = "sprint";
    if (b.simulate(q, c) && !c.attack && !c.unknownExit) out.push_back(c);
  }
}

// ============================================================================
//  SONAR PROTOCOL
//  32 bits = [type:3][payload:21][mac:8]; mac = keyed hash(type,payload,round,team)
//  A message is read one turn after it is cast at the latest, so receivers try
//  the current and the previous round.
// ============================================================================
enum : u32 { T_SYM = 1, T_PORTAL = 2, T_ENEMY = 3, T_KELP = 4, T_BIG = 5 };

u32 Bot::mac(u32 type, u32 payload, int r) const {
  return mix32(type * 0x9E3779B1u ^ payload * 0x85EBCA77u ^ (u32)r * 0xC2B2AE3Du ^ salt) & 0xFFu;
}

void Bot::readSonar() {
  for (u32 m : inbox) {
    u32 type = m >> 29, payload = (m >> 8) & 0x1FFFFFu, tag = m & 0xFFu;
    if (tag != mac(type, payload, round) && tag != mac(type, payload, round - 1)) continue;  // not ours
    switch (type) {
      case T_SYM: {
        int s = (int)(payload & 3u);
        if (payload & 128u) aggrHeard = ((payload >> 3) & 15u) / 15.0;
        if ((payload & 256u) && aggrHit < 1.0) aggrHit = 1.0;   // a teammate saw strikes
        if (s >= 1 && s <= 3 && !symKnown && gBad[s] == 0) {
          symKnown = s; symHeard = true;
          cdShared = ((payload >> 2) & 1u) && cBad[s] == 0;
          applySymmetryToAll();
        }
        break;
      }
      case T_PORTAL: {
        int o = (int)(payload & 1u), x = (int)((payload >> 1) & 63u), y = (int)((payload >> 7) & 63u);
        int hsh = (int)((payload >> 13) & 0xFFu);
        if (x >= W || y >= H) break;
        int t = tileOf(x, y);
        if (ek[o][t] == K_UNK) { ek[o][t] = K_PORT; dirtyEdge(o, t); }
        if (ek[o][t] != K_PORT) break;  // contradicts our own eyes
        bool dup = false;
        for (auto& c : pclaims) if (c.o == o && c.t == t) dup = true;
        if (!dup && pclaims.size() < 256) pclaims.push_back({hsh, o, t});
        if (epart[o][t] < 0)
          for (auto& p : pobs)
            if (p.o == o && p.t != t && (p.id & 0xFF) == hsh && epart[o][p.t] < 0) { pairPortals(o, p.t, t); break; }
        break;
      }
      case T_ENEMY: {
        int x = (int)(payload & 63u), y = (int)((payload >> 6) & 63u), len = (int)((payload >> 12) & 511u);
        if (x < W && y < H) {
          if (sightings.size() > 32) sightings.erase(sightings.begin());
          sightings.push_back({tileOf(x, y), len, round});
        }
        break;
      }
      case T_KELP: {
        int o = (int)(payload & 1u), x = (int)((payload >> 1) & 63u), y = (int)((payload >> 7) & 63u);
        u32 mask = (payload >> 13) & 0xFFu;
        if (x >= W || y >= H) break;
        for (int k = 0; k < 8; k++) {
          int tt = (o == 0) ? tileOf((x + k) % W, y) : tileOf(x, (y + k) % H);
          if (ek[o][tt] == K_UNK) recordEdge(o, tt, (mask >> k) & 1u ? K_KELP : K_OPEN, -1, false);
        }
        break;
      }
      case T_BIG: {
        int x = (int)(payload & 63u), y = (int)((payload >> 6) & 63u), len = (int)((payload >> 12) & 511u);
        if (x < W && y < H) { friendBigT = tileOf(x, y); friendBigLen = len; friendBigR = round; }
        break;
      }
      default: break;
    }
  }
}

bool Bot::chooseSonar(u32& out) {
  auto enc = [&](u32 type, u32 payload) {
    payload &= 0x1FFFFFu;
    return (type << 29) | (payload << 8) | mac(type, payload, round);
  };
  // time-critical intel first: a big enemy head in view
  int bestLen = 0, bt = -1;
  for (auto& o : others)
    if (o.enemy && o.head >= 0 && o.lenLow > bestLen) { bestLen = o.lenLow; bt = o.head; }
  if (bt >= 0 && bestLen >= 6 && (round & 1)) {
    out = enc(T_ENEMY, (u32)(bt % W) | ((u32)(bt / W) << 6) | ((u32)std::min(bestLen, 511) << 12));
    return true;
  }
  int slot = (round + myId) % 5;
  for (int tries = 0; tries < 5; tries++, slot = (slot + 1) % 5) {
    switch (slot) {
      case 0:
        if (symKnown || aggrPass + aggrHit >= 3) {
          u32 ag = (u32)std::lround(aggression() * 15.0);
          out = enc(T_SYM, (u32)symKnown | (cdShared ? 4u : 0u) | (ag << 3) | ((aggrPass + aggrHit >= 3) ? 128u : 0u) |
                               (kamikazeEvidence() ? 256u : 0u));
          return true;
        }
        break;
      case 1: {
        int n = (int)pobs.size();
        if (!n) break;
        auto& p = pobs[(msgCursor++) % n];
        out = enc(T_PORTAL, (u32)p.o | ((u32)(p.t % W) << 1) | ((u32)(p.t / W) << 7) | ((u32)(p.id & 0xFF) << 13));
        return true;
      }
      case 2: {
        for (int tr = 0; tr < 24; tr++) {
          msgCursor++;
          int t = (int)(((u32)msgCursor * 2654435761u) % (u32)NT);
          int o = msgCursor & 1;
          u32 mask = 0; bool ok = true; int kel = 0;
          for (int k = 0; k < 8; k++) {
            int tt = (o == 0) ? tileOf((t % W + k) % W, t / W) : tileOf(t % W, (t / W + k) % H);
            int8_t e = ek[o][tt];
            if (e == K_UNK || e == K_PORT) { ok = false; break; }
            if (e == K_KELP) { mask |= 1u << k; kel++; }
          }
          if (ok && kel > 0) {
            out = enc(T_KELP, (u32)o | ((u32)(t % W) << 1) | ((u32)(t / W) << 7) | (mask << 13));
            return true;
          }
        }
        break;
      }
      case 3:
        if (bt >= 0 && bestLen >= 3) {
          out = enc(T_ENEMY, (u32)(bt % W) | ((u32)(bt / W) << 6) | ((u32)std::min(bestLen, 511) << 12));
          return true;
        }
        break;
      case 4:
        if (anchorNow() || length >= 20) {
          out = enc(T_BIG, (u32)(headT % W) | ((u32)(headT / W) << 6) | ((u32)std::min(length, 511) << 12));
          return true;
        }
        break;
    }
  }
  return false;
}

// ============================================================================
//  THE TURN
// ============================================================================
std::string Bot::act() {
  t0 = nowNs();
  integrate();
  updateTrail();
  readSonar();
  buildNeighbors();
  buildOccupancy();
  buildDistances();
  buildPearlValues();
  if (gpow[0] == 0.f) {
    double g = 1.0;
    for (size_t i = 0; i < gpow.size(); i++) { gpow[i] = (float)g; g *= P.gamma; }
  }

  std::vector<Cand> cands;
  genCandidates(cands);
  int bestI = -1;
  double bestSc = -1e18;
  for (int i = 0; i < (int)cands.size(); i++) {
    if (cands[i].attack) continue;
    evaluate(cands[i]);
    if (cands[i].score > bestSc) { bestSc = cands[i].score; bestI = i; }
  }
  bool trapped = bestI < 0 || bestSc < -W_TRAP * 0.5;
  // Paying length to dodge a strike is only worth it for big dragons: small
  // ones need every segment to keep splitting (the unit war).
  bool menaced = (anchorNow() || length >= P.sprintMinLen) && kamikazeEvidence() && bestI >= 0 &&
                 cands[bestI].dest >= 0 && threat[cands[bestI].dest] * W_RISK > P.threatSprint;
  // Escape sprints: pay length to leave a closing pocket or a strike zone.
  if ((trapped || menaced) && length >= 3 && nowNs() - t0 < 30000000LL) {
    std::vector<Cand> sp;
    genSprints(*this, sp, std::min(3, length - 1));
    for (auto& c : sp) {
      if (nowNs() - t0 > 45000000LL) break;  // stay far below the 100M-point turn limit
      evaluate(c);
      cands.push_back(c);
      if (c.score > bestSc) { bestSc = c.score; bestI = (int)cands.size() - 1; }
    }
    trapped = bestI < 0 || bestSc < -W_TRAP * 0.5;
  }
  // Escape split: the long rear half leaves through the old tail.
  int escapeSplit = 0;
  int escLimit = length >= P.decoyLen ? unitLimit : unitLimit - P.reserveSmall;
  if (trapped && length >= 4 && unitCount < escLimit && bodyKnown == length) {
    for (int n = length - 2; n >= 2; n--) {
      int bestSp = childRoom(n, 2 * n + 10);
      if (bestSp >= n + 2) { escapeSplit = n; break; }
      if (nowNs() - t0 > 40000000LL) break;
    }
  }
  // Decoy split: a big dragon whose best move still leaves its head inside an
  // enemy's strike range gives the enemy only a tiny front part to hit.  The
  // bulk (our rear segments) becomes a new dragon at our tail, facing away.
  int decoySplit = 0;
  if (!trapped && !escapeSplit && bestI >= 0 && length >= P.decoyLen && unitCount < unitLimit &&
      bodyKnown == length && cands[bestI].dest >= 0 && !cands[bestI].unknownExit) {
    double thrBest = threat[cands[bestI].dest] * W_RISK;
    if (thrBest >= P.decoyThr && threatTail <= 0.f) {
      for (int n = length - 2; n >= std::max(2, length / 2); n--) {
        int room = childRoom(n, 2 * n + 10);
        if (room >= n + 2) {
          // what the small front part still risks where it stands
          double pw = P.unitK / (unitCount + 1.0) + lenWorth(length - n);
          double parentRisk = myWorth() > 0 ? threatHere * pw / myWorth() : 0.0;
          if (parentRisk + 1.0 < thrBest) decoySplit = n;
          break;
        }
        if (nowNs() - t0 > 40000000LL) break;
      }
    }
  }
  // Strikes: head-to-head into bigger enemies (or anything if doomed).
  int strikeI = -1;
  double strikeSc = -1e18;
  const char* strikeWhy = "strike";
  for (int i = 0; i < (int)cands.size(); i++) {
    Cand& c = cands[i];
    if (!c.attack) continue;
    const Other* o = nullptr;
    for (auto& q : others) if (q.id == c.attackId) o = &q;
    if (!o) continue;
    double mine = length, theirs = o->lenLow;
    // trade when what they lose is worth more than what we lose
    const char* r = nullptr;
    if (P.strikeGain * enemyWorth(o->lenLow) >= myWorth() + 1.0) r = "strike-value";
    if (anchorNow() && length >= 8) r = nullptr;
    if (!anchorNow()) {
      for (auto& f : others)
        if (!f.enemy && f.head >= 0 && f.lenLow >= std::max(10, 2 * length) && chebyshev(f.head, o->head) <= 3 &&
            f.lenLow > theirs)
          r = "strike-guard";
      if (round >= 470 && theirs >= 0.6 * mine) r = "strike-late";
    }
    // A strike kills us too: with one dragon left that is elimination.
    if (unitCount <= 1) r = nullptr;
    if (trapped && !escapeSplit) r = "strike-doomed";
    if (trapped && escapeSplit && theirs >= 0.5 * mine) r = "strike-doomed";
    if (!r) continue;
    double sc = 1000 + theirs - mine - 0.1 * (double)c.steps.size();
    if (sc > strikeSc) { strikeSc = sc; strikeI = i; strikeWhy = r; }
  }

  std::string out;
  out.reserve(96);
  bool decided = false;
  const char* why = "step";
  if (strikeI >= 0) {
    Cand& c = cands[strikeI];
    out = "MOVE " + c.steps + "\n";
    lastKind = 0; lastPath = c.path; expectHead = -1;
    decided = true; why = strikeWhy;
  }
  if (!decided && escapeSplit) {
    out = "SPLIT " + std::to_string(escapeSplit) + "\n";
    lastKind = 1; lastSplit = escapeSplit;
    decided = true; why = "esplit";
  }
  if (!decided && decoySplit) {
    out = "SPLIT " + std::to_string(decoySplit) + "\n";
    lastKind = 1; lastSplit = decoySplit;
    decided = true; why = "dsplit";
  }
  // Growth split: multiply while the arena can feed us.
  if (!decided && !trapped && round >= 3 && round <= P.lateNoSplit && bodyKnown == length && unitCount < unitLimit) {
    int cap = unitCap();
    bool want = unitCount < std::min(cap, unitLimit - P.reserve) && length >= splitLen() && length < P.growLen;
    if (isAnchor) want = want && unitCount < std::min(cap, P.anchorSeed) && round <= P.anchorSeedRound;
    double q;
    if (want && nowNs() - t0 < 40000000LL) {
      int n = length / 2;
      if (splitSafe(n, q)) {
        out = "SPLIT " + std::to_string(n) + "\n";
        lastKind = 1; lastSplit = n;
        decided = true; why = "gsplit";
      }
    }
  }
  if (!decided && bestI >= 0) {
    Cand& c = cands[bestI];
    out = "MOVE " + c.steps + "\n";
    lastKind = 0; lastPath = c.path;
    expectHead = c.path.empty() ? -1 : c.path.back();
    if (expectHead < 0) lastPath.clear();
    decided = true;
    why = c.unknownExit ? "blindportal" : trapped ? "trapmove" : (c.steps.size() > 1 ? "sprint" : "step");
  }
  if (!decided) {
    // Nothing survives: split so part of us lives, else accept fate.
    if (length >= 4 && unitCount < escLimit) {
      out = "SPLIT " + std::to_string(length - 2) + "\n";
      lastKind = 1; lastSplit = length - 2;
      why = "lastsplit";
    } else {
      out = std::string("MOVE ") + DCH[facing] + "\n";
      lastKind = 0; lastPath.clear(); expectHead = -1;
      why = "fallback";
    }
  }
  u32 msg;
  if (chooseSonar(msg)) out += "SONAR " + std::to_string(msg) + "\n";
  (void)why;  // decision label; print it in an INDICATOR line to see it in replays
  out += "ENDTURN\n";
  return out;
}

#undef W_FIELD
#undef W_EAT
#undef W_TRAP
#undef W_SPACE
#undef W_TERR
#undef W_RISK
#undef W_LEN
#undef W_CONTEST
}  // namespace seadragon

// ============================================================================
//  ENTRY POINT: one process per dragon; exits when the engine closes stdin.
// ============================================================================
int main() {
  static char outbuf[1 << 16];
  setvbuf(stdout, outbuf, _IOFBF, sizeof outbuf);  // one write per turn
  static seadragon::Bot bot;
  static char line[1 << 16];
  while (fgets(line, sizeof line, stdin)) {
    if (bot.feed(line)) {
      std::string r = bot.act();
      fwrite(r.data(), 1, r.size(), stdout);
      fflush(stdout);
    }
    if (bot.finished) break;
  }
  return 0;
}