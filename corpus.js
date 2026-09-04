// The corpus this floor replays: pre-written code, tokenised once at load.
// Nothing is generated at runtime — the sessions only decide *when* each
// token lands, so the wall shows what a given tok/s budget really looks like.
const SNIPPETS = [
{t:"LRU cache with TTL", code:`class LRUCache:
    """Fixed-capacity cache with per-entry expiry."""

    def __init__(self, capacity: int, ttl: float = 60.0):
        self.capacity = capacity
        self.ttl = ttl
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key):
        entry = self._data.get(key)
        if entry is None:
            self.misses += 1
            return None
        expires_at, value = entry
        if expires_at <= time.monotonic():
            del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (time.monotonic() + self.ttl, value)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)
            self.evictions += 1
`},
{t:"token-bucket rate limiter", code:`class TokenBucket:
    """Smooth burst control: refill continuously, spend on demand."""

    def __init__(self, rate: float, burst: float):
        self.rate = rate
        self.burst = burst
        self._tokens = burst
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        delta = now - self._last
        self._last = now
        self._tokens = min(self.burst, self._tokens + delta * self.rate)

    def take(self, n: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def wait_time(self, n: float = 1.0) -> float:
        with self._lock:
            self._refill()
            if self._tokens >= n:
                return 0.0
            return (n - self._tokens) / self.rate
`},
{t:"streaming JSON tokenizer", code:`def tokenize(stream):
    """Yield JSON tokens without holding the whole document."""
    buf = []
    in_string = False
    escaped = False
    for chunk in stream:
        for ch in chunk:
            if in_string:
                buf.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                    yield ("str", "".join(buf))
                    buf = []
                continue
            if ch == '"':
                in_string = True
                buf = [ch]
            elif ch in "{}[],:":
                if buf:
                    yield classify("".join(buf))
                    buf = []
                yield ("punct", ch)
            elif ch.isspace():
                if buf:
                    yield classify("".join(buf))
                    buf = []
            else:
                buf.append(ch)
    if buf:
        yield classify("".join(buf))
`},
{t:"trie with prefix walk", code:`struct Node {
    children: HashMap<char, Node>,
    terminal: bool,
}

impl Node {
    fn new() -> Self {
        Node { children: HashMap::new(), terminal: false }
    }
}

pub struct Trie { root: Node, len: usize }

impl Trie {
    pub fn insert(&mut self, word: &str) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_insert_with(Node::new);
        }
        if !node.terminal {
            node.terminal = true;
            self.len += 1;
        }
    }

    pub fn starts_with(&self, prefix: &str) -> Vec<String> {
        let mut node = &self.root;
        for ch in prefix.chars() {
            match node.children.get(&ch) {
                Some(next) => node = next,
                None => return Vec::new(),
            }
        }
        let mut out = Vec::new();
        walk(node, &mut prefix.to_string(), &mut out);
        out
    }
}
`},
{t:"binary heap priority queue", code:`export class PriorityQueue {
  #heap = [];
  #cmp;

  constructor(cmp = (a, b) => a - b) {
    this.#cmp = cmp;
  }

  get size() { return this.#heap.length; }

  push(item) {
    const h = this.#heap;
    h.push(item);
    let i = h.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.#cmp(h[i], h[parent]) >= 0) break;
      [h[i], h[parent]] = [h[parent], h[i]];
      i = parent;
    }
  }

  pop() {
    const h = this.#heap;
    if (h.length === 0) return undefined;
    const top = h[0];
    const last = h.pop();
    if (h.length) { h[0] = last; this.#sink(0); }
    return top;
  }
}
`},
{t:"minimal HTTP/1.1 request", code:`func Fetch(ctx context.Context, addr, path string) ([]byte, error) {
	conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
	if err != nil {
		return nil, fmt.Errorf("dial %s: %w", addr, err)
	}
	defer conn.Close()

	req := "GET " + path + " HTTP/1.1\\r\\n" +
		"Host: " + addr + "\\r\\n" +
		"Connection: close\\r\\n" +
		"Accept-Encoding: identity\\r\\n\\r\\n"

	if _, err := io.WriteString(conn, req); err != nil {
		return nil, err
	}

	reader := bufio.NewReader(conn)
	status, err := reader.ReadString('\\n')
	if err != nil {
		return nil, err
	}
	if !strings.HasPrefix(status, "HTTP/1.1 200") {
		return nil, fmt.Errorf("bad status: %s", strings.TrimSpace(status))
	}
	for {
		line, err := reader.ReadString('\\n')
		if err != nil || line == "\\r\\n" {
			break
		}
	}
	return io.ReadAll(reader)
}
`},
{t:"CSV reader with quotes", code:`def read_rows(text: str, delimiter: str = ","):
    """RFC 4180-ish reader: quotes, escaped quotes, embedded newlines."""
    row, field, quoted, i = [], [], False, 0
    while i < len(text):
        ch = text[i]
        if quoted:
            if ch == '"':
                if i + 1 < len(text) and text[i + 1] == '"':
                    field.append('"')
                    i += 1
                else:
                    quoted = False
            else:
                field.append(ch)
        elif ch == '"':
            quoted = True
        elif ch == delimiter:
            row.append("".join(field))
            field = []
        elif ch == "\\n":
            row.append("".join(field))
            yield row
            row, field = [], []
        elif ch != "\\r":
            field.append(ch)
        i += 1
    if field or row:
        row.append("".join(field))
        yield row
`},
{t:"LL(1) arithmetic parser", code:`class Parser:
    """expr := term (('+'|'-') term)*   term := atom (('*'|'/') atom)*"""

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def eat(self, kind):
        tok = self.peek()
        if tok is None or tok.kind != kind:
            raise SyntaxError(f"expected {kind}, got {tok!r} at {self.pos}")
        self.pos += 1
        return tok

    def expr(self):
        node = self.term()
        while (tok := self.peek()) and tok.value in "+-":
            self.pos += 1
            node = BinOp(tok.value, node, self.term())
        return node

    def atom(self):
        tok = self.peek()
        if tok.kind == "number":
            self.pos += 1
            return Num(float(tok.value))
        self.eat("lparen")
        node = self.expr()
        self.eat("rparen")
        return node
`},
{t:"bloom filter", code:`class BloomFilter:
    """Probabilistic membership: no false negatives, tunable false positives."""

    def __init__(self, n: int, p: float = 0.01):
        self.m = math.ceil(-n * math.log(p) / (math.log(2) ** 2))
        self.k = max(1, round(self.m / n * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)
        self.count = 0

    def _hashes(self, item: bytes):
        h1 = int.from_bytes(hashlib.blake2b(item, digest_size=8).digest(), "big")
        h2 = int.from_bytes(hashlib.blake2s(item, digest_size=8).digest(), "big") | 1
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item: bytes):
        for bit in self._hashes(item):
            self.bits[bit >> 3] |= 1 << (bit & 7)
        self.count += 1

    def __contains__(self, item: bytes) -> bool:
        return all(self.bits[b >> 3] & (1 << (b & 7)) for b in self._hashes(item))
`},
{t:"interval tree", code:`class IntervalTree:
    """Augmented BST: every node caches the max endpoint of its subtree."""

    def insert(self, lo, hi, payload=None):
        self.root = self._insert(self.root, Interval(lo, hi, payload))
        self.size += 1

    def _insert(self, node, iv):
        if node is None:
            return iv
        if iv.lo < node.lo:
            node.left = self._insert(node.left, iv)
        else:
            node.right = self._insert(node.right, iv)
        node.max_hi = max(node.max_hi, iv.hi)
        return node

    def overlapping(self, lo, hi):
        out = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is None or node.max_hi < lo:
                continue
            if node.lo <= hi and lo <= node.hi:
                out.append(node)
            stack.append(node.left)
            if node.lo <= hi:
                stack.append(node.right)
        return out
`},
{t:"Dijkstra shortest path", code:`func ShortestPath(g Graph, src, dst int) ([]int, float64) {
	dist := make(map[int]float64, len(g))
	prev := make(map[int]int, len(g))
	for node := range g {
		dist[node] = math.Inf(1)
	}
	dist[src] = 0

	pq := &Heap{{node: src, cost: 0}}
	heap.Init(pq)

	for pq.Len() > 0 {
		cur := heap.Pop(pq).(Item)
		if cur.node == dst {
			break
		}
		if cur.cost > dist[cur.node] {
			continue
		}
		for _, edge := range g[cur.node] {
			alt := cur.cost + edge.weight
			if alt < dist[edge.to] {
				dist[edge.to] = alt
				prev[edge.to] = cur.node
				heap.Push(pq, Item{node: edge.to, cost: alt})
			}
		}
	}
	return reconstruct(prev, src, dst), dist[dst]
}
`},
{t:"consistent hashing ring", code:`class HashRing:
    """Virtual nodes keep the key distribution even when servers move."""

    def __init__(self, nodes=(), replicas: int = 160):
        self.replicas = replicas
        self._ring: dict[int, str] = {}
        self._sorted: list[int] = []
        for node in nodes:
            self.add(node)

    def _hash(self, key: str) -> int:
        return int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "big")

    def add(self, node: str):
        for i in range(self.replicas):
            self._ring[self._hash(f"{node}#{i}")] = node
        self._sorted = sorted(self._ring)

    def remove(self, node: str):
        for i in range(self.replicas):
            self._ring.pop(self._hash(f"{node}#{i}"), None)
        self._sorted = sorted(self._ring)

    def get(self, key: str) -> str:
        if not self._ring:
            raise KeyError("empty ring")
        idx = bisect.bisect(self._sorted, self._hash(key)) % len(self._sorted)
        return self._ring[self._sorted[idx]]
`},
{t:"topological sort", code:`def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Kahn's algorithm. Raises on cycles, names the nodes involved."""
    indegree = {node: 0 for node in graph}
    for deps in graph.values():
        for dep in deps:
            indegree[dep] = indegree.get(dep, 0) + 1

    queue = deque(sorted(n for n, d in indegree.items() if d == 0))
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for dep in graph.get(node, ()):
            indegree[dep] -= 1
            if indegree[dep] == 0:
                queue.append(dep)

    if len(order) != len(indegree):
        stuck = sorted(n for n, d in indegree.items() if d > 0)
        raise ValueError(f"cycle through: {', '.join(stuck)}")
    return order
`},
{t:"run-length + Huffman", code:`def rle_encode(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(data):
        run = 1
        while i + run < len(data) and data[i + run] == data[i] and run < 255:
            run += 1
        out.append(run)
        out.append(data[i])
        i += run
    return bytes(out)


def huffman_table(freqs: dict[int, int]) -> dict[int, str]:
    heap = [(count, i, Leaf(sym)) for i, (sym, count) in enumerate(freqs.items())]
    heapq.heapify(heap)
    counter = len(heap)
    while len(heap) > 1:
        c1, _, left = heapq.heappop(heap)
        c2, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (c1 + c2, counter, Branch(left, right)))
        counter += 1
    codes: dict[int, str] = {}
    assign(heap[0][2], "", codes)
    return codes
`},
{t:"append-only key-value store", code:`impl Store {
    /// Append-only log with an in-memory offset index. Crash-safe by design.
    pub fn set(&mut self, key: &str, value: &[u8]) -> io::Result<()> {
        let offset = self.log.stream_position()?;
        let header = Header { klen: key.len() as u32, vlen: value.len() as u32 };
        self.log.write_all(&header.to_bytes())?;
        self.log.write_all(key.as_bytes())?;
        self.log.write_all(value)?;
        self.log.flush()?;
        self.index.insert(key.to_string(), offset);
        self.dirty += 1;
        if self.dirty > self.compact_after {
            self.compact()?;
        }
        Ok(())
    }

    pub fn get(&mut self, key: &str) -> io::Result<Option<Vec<u8>>> {
        let Some(&offset) = self.index.get(key) else { return Ok(None) };
        self.log.seek(SeekFrom::Start(offset))?;
        let header = Header::read(&mut self.log)?;
        self.log.seek(SeekFrom::Current(header.klen as i64))?;
        let mut buf = vec![0u8; header.vlen as usize];
        self.log.read_exact(&mut buf)?;
        Ok(Some(buf))
    }
}
`},
{t:"A* pathfinder on a grid", code:`function findPath(grid, start, goal) {
  const open = new PriorityQueue((a, b) => a.f - b.f);
  const gScore = new Map([[key(start), 0]]);
  const cameFrom = new Map();
  open.push({ ...start, f: heuristic(start, goal) });

  while (open.size) {
    const cur = open.pop();
    if (cur.x === goal.x && cur.y === goal.y) {
      return reconstruct(cameFrom, cur);
    }
    for (const next of neighbours(grid, cur)) {
      const tentative = gScore.get(key(cur)) + cost(grid, next);
      if (tentative < (gScore.get(key(next)) ?? Infinity)) {
        cameFrom.set(key(next), cur);
        gScore.set(key(next), tentative);
        open.push({ ...next, f: tentative + heuristic(next, goal) });
      }
    }
  }
  return null; // unreachable
}
`},
{t:"reliable transport over UDP", code:`class Session:
    """Sequence numbers, cumulative acks, exponential backoff."""

    async def send(self, payload: bytes):
        seq = self.next_seq
        self.next_seq += 1
        packet = struct.pack("!IHH", seq, len(payload), crc16(payload)) + payload
        self.unacked[seq] = (packet, time.monotonic(), 0)
        self.transport.sendto(packet, self.peer)

    async def _retransmit_loop(self):
        while not self.closed:
            now = time.monotonic()
            for seq, (packet, sent_at, tries) in list(self.unacked.items()):
                timeout = self.rto * (2 ** tries)
                if now - sent_at < timeout:
                    continue
                if tries >= self.max_retries:
                    self.unacked.pop(seq, None)
                    self.on_drop(seq)
                    continue
                self.transport.sendto(packet, self.peer)
                self.unacked[seq] = (packet, now, tries + 1)
            await asyncio.sleep(0.005)
`},
{t:"Myers diff", code:`def diff(a: list[str], b: list[str]):
    """Shortest edit script, O((N+M)D)."""
    n, m = len(a), len(b)
    maxd = n + m
    v = {1: 0}
    trace = []
    for d in range(maxd + 1):
        trace.append(dict(v))
        for k in range(-d, d + 1, 2):
            if k == -d or (k != d and v[k - 1] < v[k + 1]):
                x = v[k + 1]
            else:
                x = v[k - 1] + 1
            y = x - k
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1
            v[k] = x
            if x >= n and y >= m:
                return backtrack(trace, a, b)
    raise RuntimeError("unreachable")
`},
{t:"double-entry ledger", code:`@dataclass(frozen=True)
class Entry:
    account: str
    amount: Decimal
    currency: str = "EUR"


class Ledger:
    """Every transaction must balance to zero before it is written."""

    def post(self, entries: list[Entry], memo: str = "") -> str:
        totals: dict[str, Decimal] = defaultdict(Decimal)
        for entry in entries:
            totals[entry.currency] += entry.amount
        for currency, total in totals.items():
            if total != 0:
                raise Unbalanced(f"{currency} off by {total}")

        txid = uuid.uuid4().hex
        with self.conn:
            for entry in entries:
                self.conn.execute(
                    "INSERT INTO entries (txid, account, amount, currency, memo)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (txid, entry.account, str(entry.amount), entry.currency, memo),
                )
        return txid
`},
{t:"Roman numerals", code:`const NUMERALS = [
  [1000, "M"], [900, "CM"], [500, "D"], [400, "CD"],
  [100, "C"], [90, "XC"], [50, "L"], [40, "XL"],
  [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"],
];

export function toRoman(n) {
  if (!Number.isInteger(n) || n < 1 || n > 3999) {
    throw new RangeError("out of range: " + n);
  }
  let out = "";
  for (const [value, symbol] of NUMERALS) {
    while (n >= value) { out += symbol; n -= value; }
  }
  return out;
}

export function fromRoman(s) {
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    const cur = VALUES[s[i]], next = VALUES[s[i + 1]] ?? 0;
    total += cur < next ? -cur : cur;
  }
  return total;
}
`},
{t:"LZ77 sliding window", code:`def compress(data: bytes, window: int = 4096, look: int = 18):
    out = bytearray()
    i = 0
    while i < len(data):
        best_len, best_dist = 0, 0
        start = max(0, i - window)
        for j in range(start, i):
            length = 0
            while (length < look and i + length < len(data)
                   and data[j + length] == data[i + length]):
                length += 1
            if length > best_len:
                best_len, best_dist = length, i - j
        if best_len >= 3:
            out.append(0x80 | (best_len - 3))
            out += best_dist.to_bytes(2, "big")
            i += best_len
        else:
            out.append(0x00)
            out.append(data[i])
            i += 1
    return bytes(out)
`},
{t:"k-means clustering", code:`def kmeans(points: np.ndarray, k: int, iters: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    centroids = points[rng.choice(len(points), k, replace=False)]

    for step in range(iters):
        distances = np.linalg.norm(points[:, None] - centroids[None], axis=2)
        labels = distances.argmin(axis=1)

        moved = 0.0
        for c in range(k):
            members = points[labels == c]
            if len(members) == 0:
                members = points[rng.integers(len(points))][None]
            new = members.mean(axis=0)
            moved = max(moved, float(np.linalg.norm(new - centroids[c])))
            centroids[c] = new

        if moved < 1e-6:
            break
    inertia = float(((points - centroids[labels]) ** 2).sum())
    return centroids, labels, inertia
`},
{t:"cron expression matcher", code:`class CronExpr:
    FIELDS = ("minute", "hour", "day", "month", "weekday")
    RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))

    def __init__(self, spec: str):
        parts = spec.split()
        if len(parts) != 5:
            raise ValueError(f"expected 5 fields, got {len(parts)}: {spec!r}")
        self.sets = [self._parse(p, lo, hi) for p, (lo, hi) in zip(parts, self.RANGES)]

    def _parse(self, part: str, lo: int, hi: int) -> set[int]:
        values: set[int] = set()
        for chunk in part.split(","):
            step = 1
            if "/" in chunk:
                chunk, raw = chunk.split("/", 1)
                step = int(raw)
            if chunk in ("*", ""):
                start, end = lo, hi
            elif "-" in chunk:
                a, b = chunk.split("-", 1)
                start, end = int(a), int(b)
            else:
                start = end = int(chunk)
            values.update(range(start, end + 1, step))
        return values
`},
{t:"base64 codec", code:`static const char ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t b64_encode(const uint8_t *in, size_t len, char *out) {
    size_t o = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t chunk = in[i] << 16;
        if (i + 1 < len) chunk |= in[i + 1] << 8;
        if (i + 2 < len) chunk |= in[i + 2];

        out[o++] = ALPHABET[(chunk >> 18) & 0x3F];
        out[o++] = ALPHABET[(chunk >> 12) & 0x3F];
        out[o++] = (i + 1 < len) ? ALPHABET[(chunk >> 6) & 0x3F] : '=';
        out[o++] = (i + 2 < len) ? ALPHABET[chunk & 0x3F] : '=';
    }
    out[o] = '\\0';
    return o;
}
`},
{t:"backtracking sudoku solver", code:`def solve(board: list[list[int]]) -> bool:
    """Fill the first empty cell with every legal digit, recurse, undo."""
    spot = find_empty(board)
    if spot is None:
        return True
    row, col = spot

    for digit in range(1, 10):
        if not is_legal(board, row, col, digit):
            continue
        board[row][col] = digit
        if solve(board):
            return True
        board[row][col] = 0
    return False


def is_legal(board, row, col, digit) -> bool:
    if digit in board[row]:
        return False
    if any(board[r][col] == digit for r in range(9)):
        return False
    br, bc = 3 * (row // 3), 3 * (col // 3)
    return all(
        board[r][c] != digit
        for r in range(br, br + 3)
        for c in range(bc, bc + 3)
    )
`},
{t:"brainfuck interpreter", code:`pub fn run(src: &str, input: &[u8]) -> Vec<u8> {
    let program: Vec<char> = src.chars().filter(|c| "><+-.,[]".contains(*c)).collect();
    let jumps = build_jump_table(&program);

    let mut tape = vec![0u8; 30_000];
    let mut ptr = 0usize;
    let mut pc = 0usize;
    let mut cursor = 0usize;
    let mut out = Vec::new();

    while pc < program.len() {
        match program[pc] {
            '>' => ptr = (ptr + 1) % tape.len(),
            '<' => ptr = (ptr + tape.len() - 1) % tape.len(),
            '+' => tape[ptr] = tape[ptr].wrapping_add(1),
            '-' => tape[ptr] = tape[ptr].wrapping_sub(1),
            '.' => out.push(tape[ptr]),
            ',' => { tape[ptr] = input.get(cursor).copied().unwrap_or(0); cursor += 1; }
            '[' => if tape[ptr] == 0 { pc = jumps[&pc]; },
            ']' => if tape[ptr] != 0 { pc = jumps[&pc]; },
            _ => unreachable!(),
        }
        pc += 1;
    }
    out
}
`},
{t:"ray-sphere intersection", code:`Hit trace(const Ray &ray, const Scene &scene, int depth) {
    Hit best{.t = INFINITY, .hit = false};
    for (const Sphere &s : scene.spheres) {
        Vec3 oc = ray.origin - s.center;
        float b = 2.0f * dot(oc, ray.dir);
        float c = dot(oc, oc) - s.radius * s.radius;
        float disc = b * b - 4.0f * c;
        if (disc < 0.0f) continue;

        float t = (-b - sqrtf(disc)) * 0.5f;
        if (t > EPSILON && t < best.t) {
            best.t = t;
            best.hit = true;
            best.point = ray.at(t);
            best.normal = normalize(best.point - s.center);
            best.material = s.material;
        }
    }
    if (!best.hit || depth <= 0) return best;
    return shade(best, scene, depth - 1);
}
`},
{t:"WebSocket frame decoder", code:`export function decodeFrame(buf) {
  if (buf.length < 2) return null;
  const fin = (buf[0] & 0x80) !== 0;
  const opcode = buf[0] & 0x0f;
  const masked = (buf[1] & 0x80) !== 0;
  let len = buf[1] & 0x7f;
  let offset = 2;

  if (len === 126) {
    len = buf.readUInt16BE(offset);
    offset += 2;
  } else if (len === 127) {
    len = Number(buf.readBigUInt64BE(offset));
    offset += 8;
  }

  let mask;
  if (masked) {
    mask = buf.subarray(offset, offset + 4);
    offset += 4;
  }
  if (buf.length < offset + len) return null; // need more bytes

  const payload = Buffer.from(buf.subarray(offset, offset + len));
  if (masked) {
    for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
  }
  return { fin, opcode, payload, rest: buf.subarray(offset + len) };
}
`},
{t:"deploy.sh", code:`#!/usr/bin/env bash
set -euo pipefail

APP=lucille
REMOTE=deploy@edge-01
RELEASES=/srv/$APP/releases
STAMP=$(date +%Y%m%d%H%M%S)

log() { printf '[%s] %s\n' "$(date +%T)" "$*" >&2; }

log "building $APP"
cargo build --release --locked

log "packing"
tar -czf "/tmp/$APP-$STAMP.tgz" -C target/release "$APP"

log "shipping to $REMOTE"
scp -q "/tmp/$APP-$STAMP.tgz" "$REMOTE:$RELEASES/"

ssh "$REMOTE" bash -s <<EOF
  set -euo pipefail
  cd "$RELEASES"
  mkdir -p "$STAMP" && tar -xzf "$APP-$STAMP.tgz" -C "$STAMP"
  ln -sfn "$RELEASES/$STAMP" /srv/$APP/current
  systemctl restart $APP
  sleep 2
  systemctl is-active --quiet $APP || { echo "restart failed"; exit 1; }
EOF

log "live: $STAMP"
`},
{t:"rotate-backups.sh", code:`#!/usr/bin/env bash
set -euo pipefail

SRC=/var/lib/postgres
DEST=/mnt/backup/pg
KEEP_DAILY=7
KEEP_WEEKLY=4

today=$(date +%F)
target="$DEST/daily/$today"

mkdir -p "$target"
pg_dumpall --clean | gzip -9 > "$target/all.sql.gz"

# checksum so a silent truncation cannot pass for a backup
sha256sum "$target/all.sql.gz" > "$target/SHA256"
size=$(stat -c %s "$target/all.sql.gz")
if [ "$size" -lt 1048576 ]; then
  echo "backup suspiciously small: $size bytes" >&2
  exit 1
fi

if [ "$(date +%u)" = "7" ]; then
  cp -al "$target" "$DEST/weekly/$today"
fi

find "$DEST/daily"  -maxdepth 1 -mindepth 1 -type d | sort | head -n -$KEEP_DAILY  | xargs -r rm -rf
find "$DEST/weekly" -maxdepth 1 -mindepth 1 -type d | sort | head -n -$KEEP_WEEKLY | xargs -r rm -rf

du -sh "$DEST"
`},
{t:"healthcheck loop", code:`#!/usr/bin/env bash
# Polls every endpoint, backs off on failure, pages once.

ENDPOINTS=(
  "https://api.internal/healthz"
  "https://gw.internal/healthz"
  "https://pi.local:8443/healthz"
)
INTERVAL=15
FAILS=0
THRESHOLD=3

notify() {
  curl -fsS -X POST "$WEBHOOK" \
    -H 'content-type: application/json' \
    -d "{\\"text\\": \\"$1\\"}" >/dev/null || true
}

trap 'echo "stopping"; exit 0' INT TERM

while true; do
  down=()
  for url in "\${ENDPOINTS[@]}"; do
    code=$(curl -o /dev/null -sw '%{http_code}' --max-time 5 "$url" || echo 000)
    [ "$code" = "200" ] || down+=("$url:$code")
  done

  if [ \${#down[@]} -eq 0 ]; then
    [ "$FAILS" -ge "$THRESHOLD" ] && notify "recovered"
    FAILS=0
  else
    FAILS=$((FAILS + 1))
    [ "$FAILS" -eq "$THRESHOLD" ] && notify "down: \${down[*]}"
  fi
  sleep "$INTERVAL"
done
`},
{t:"batch transcode", code:`#!/usr/bin/env bash
set -euo pipefail

IN=\${1:-./raw}
OUT=\${2:-./web}
JOBS=$(nproc)

mkdir -p "$OUT"

transcode() {
  local src="$1"
  local name
  name=$(basename "\${src%.*}")
  local dst="$OUT/$name.mp4"

  [ -f "$dst" ] && { echo "skip $name"; return 0; }

  ffmpeg -nostdin -loglevel error -i "$src" \
    -vf "scale='min(1920,iw)':-2" \
    -c:v libx264 -preset slow -crf 21 \
    -c:a aac -b:a 160k -movflags +faststart \
    "$dst.part"

  mv "$dst.part" "$dst"
  echo "done $name ($(du -h "$dst" | cut -f1))"
}

export -f transcode
export OUT

find "$IN" -type f \( -name '*.mov' -o -name '*.mkv' \) -print0 \
  | xargs -0 -P "$JOBS" -I{} bash -c 'transcode "$@"' _ {}
`},
{t:"pre-push hook", code:`#!/usr/bin/env bash
# .git/hooks/pre-push - refuse to push what will not build.
set -euo pipefail

remote="$1"
protected="main master release"
branch=$(git rev-parse --abbrev-ref HEAD)

for p in $protected; do
  if [ "$branch" = "$p" ]; then
    echo "refusing direct push to $p (use a branch)" >&2
    exit 1
  fi
done

if git diff --cached --name-only | grep -qE '\.env$|id_rsa|\.pem$'; then
  echo "secret-looking file staged, aborting" >&2
  exit 1
fi

echo "fmt"   && cargo fmt --check
echo "clippy" && cargo clippy --all-targets -- -D warnings
echo "test"  && cargo test --quiet

ahead=$(git rev-list --count "@{u}..HEAD" 2>/dev/null || echo 0)
echo "pushing $ahead commit(s) to $remote/$branch"
`},
{t:"tail and alert", code:`#!/usr/bin/env bash
set -uo pipefail

LOG=/var/log/gateway/app.log
WINDOW=60
LIMIT=25
declare -A seen

stdbuf -oL tail -F "$LOG" 2>/dev/null | while IFS= read -r line; do
  case "$line" in
    *ERROR*|*panic*|*FATAL*) ;;
    *) continue ;;
  esac

  now=$(date +%s)
  key=$(printf '%s' "$line" | sed -E 's/[0-9]+/N/g' | cut -c1-80)
  count=\${seen[$key]:-0}
  seen[$key]=$((count + 1))

  if [ $((count + 1)) -gt "$LIMIT" ]; then
    printf 'suppressing repeated: %s\n' "$key"
    continue
  fi

  printf '%s | %s\n' "$(date -d "@$now" +%T)" "$line"
done
`},
];
