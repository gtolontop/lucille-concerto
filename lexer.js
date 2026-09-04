// Turning source text into model-sized tokens.
//
// A real tokeniser would give pieces of ~4 characters; this one splits
// lexemes into 3-5 character chunks and tags each with what it is, so the
// wall can colour the code and advance it in believable steps.

const LEXER = (() => {
"use strict";

const KEYWORDS = new Set((
  "def class return if elif else for while import from in not and or is None True False " +
  "self async await try except finally raise with yield lambda pass break continue global " +
  "assert del as match case struct impl fn let mut pub use crate enum trait where const static " +
  "unsafe move ref type function var new export default extern package interface implements " +
  "extends throw catch this super typeof instanceof void delete do switch break case go defer " +
  "chan range map make nil err int float bool char size_t uint8_t uint32_t sizeof template " +
  "public private protected virtual override final abstract " +
  "fi then do done esac local readonly export set trap shift source until exit echo"
).split(" "));

const LEX = /(#[!\s][^\n]*|\/\/[^\n]*|\/\/\/[^\n]*|\/\*[\s\S]*?\*\/)|("""[\s\S]*?"""|"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?|`(?:\\.|[^`\\])*`)|(\b\d[\w.]*)|([A-Za-z_]\w*)|(\s+)|([^\s\w])/g;

function chunk(text, cls, out) {
  if (cls === "ws") {
    let buf = "";
    for (const ch of text) {
      if (ch === "\n") {
        if (buf) { out.push({ t: buf, c: cls }); buf = ""; }
        out.push({ t: "\n", c: cls });
      } else buf += ch;
    }
    if (buf) out.push({ t: buf, c: cls });
    return;
  }
  if (text.length <= 4) { out.push({ t: text, c: cls }); return; }
  let i = 0;
  while (i < text.length) {
    const size = 3 + ((i * 7 + text.length) % 3); // 3..5
    const piece = text.slice(i, i + size);
    const nl = piece.indexOf("\n");
    if (nl >= 0) {
      if (nl > 0) out.push({ t: piece.slice(0, nl), c: cls });
      out.push({ t: "\n", c: cls });
      i += nl + 1;
      continue;
    }
    out.push({ t: piece, c: cls });
    i += size;
  }
}

function tokenize(code) {
  const out = [];
  let m;
  LEX.lastIndex = 0;
  while ((m = LEX.exec(code)) !== null) {
    if (m[1]) chunk(m[1], "com", out);
    else if (m[2]) chunk(m[2], "str", out);
    else if (m[3]) chunk(m[3], "num", out);
    else if (m[4]) chunk(m[4], KEYWORDS.has(m[4]) ? "kw" : "ident", out);
    else if (m[5]) chunk(m[5], "ws", out);
    else chunk(m[6], "punct", out);
  }
  return out;
}

return { tokenize };
})();
