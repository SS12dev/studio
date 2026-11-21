"""Local people page: fix groups by hand, answer the review queue, mark filtered photos.

Runs on 127.0.0.1 only. Serves face crops from the workspace and downscaled copies of photos
that appear in faces.json, nothing else from disk.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from studio.faces import curate
from studio.faces.workspace import Workspace
from studio.ingest import load_rgb

CROP_RE = re.compile(r"^[0-9a-f]{16}\.jpg$")


class Faces(BaseModel):
    faces: list[str]


class Remove(Faces):
    person: str


class Move(Faces):
    src: str | None = None
    dst: str


class Training(Faces):
    exclude: bool


class Rename(BaseModel):
    old: str
    new: str


class Answer(BaseModel):
    face: str
    name: str | None = None


def create_app(ws: Workspace) -> FastAPI:
    app = FastAPI(title="Studio people")
    photo_of = {f["id"]: f["photo"] for f in ws.faces}

    def tile(fid: str, source: str, no_train: set[str], extra: dict | None = None) -> dict:
        return {"id": fid, "crop": ws.crop_name(fid), "source": source,
                "photo": Path(photo_of.get(fid, fid.split("#")[0])).name,
                "no_training": fid in no_train, **(extra or {})}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/state")
    def state() -> dict:
        no_train = curate.no_training(ws)
        people = {name: [tile(f, src, no_train) for f, src in items]
                  for name, items in sorted(curate.members(ws).items(),
                                            key=lambda kv: -len(kv[1]))}
        review = [tile(fid, "review", no_train,
                       {"guess": d["person"], "score": d["score"], "second_pass": d.get("second_pass")})
                  for fid, d in curate._decisions(ws).items() if d.get("route") == "review"]
        review.sort(key=lambda t: (t["guess"] or "", -t["score"]))
        groups = {k: [tile(f, "group", no_train) for f in v]
                  for k, v in ws.clusters().get("clusters", {}).items()
                  if k != "Unclustered" and not set(v) & {x for ids in ws.people.values() for x in ids}}
        return {"people": people, "review": review, "unnamed_groups": groups,
                "workspace": str(ws.root)}

    @app.get("/crops/{name}")
    def crop(name: str):
        if not CROP_RE.match(name):
            raise HTTPException(404)
        p = ws.crops / name
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, headers={"Cache-Control": "max-age=86400"})

    @app.get("/photo")
    def photo(face: str):
        path = photo_of.get(face)
        if not path:
            raise HTTPException(404)
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(load_rgb(Path(path), 1024)).save(buf, "JPEG", quality=85)
        return Response(buf.getvalue(), media_type="image/jpeg")

    @app.post("/api/remove")
    def remove(req: Remove) -> dict:
        curate.remove(ws, req.person, req.faces)
        return {"ok": True}

    @app.post("/api/move")
    def move(req: Move) -> dict:
        curate.assign_to(ws, req.dst.strip(), req.faces, src=req.src)
        return {"ok": True}

    @app.post("/api/training")
    def training(req: Training) -> dict:
        curate.set_training(ws, req.faces, req.exclude)
        return {"ok": True}

    @app.post("/api/rename")
    def rename(req: Rename) -> dict:
        curate.rename(ws, req.old, req.new.strip())
        return {"ok": True}

    @app.post("/api/answer")
    def answer(req: Answer) -> dict:
        curate.answer(ws, req.face, req.name)
        return {"ok": True}

    return app


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>People</title><style>
*{box-sizing:border-box}body{margin:0;font:14px -apple-system,system-ui,sans-serif;background:#111;color:#eee;display:flex;height:100vh}
nav{width:240px;border-right:1px solid #333;overflow:auto;padding:12px}nav h3{font-size:12px;color:#888;text-transform:uppercase;margin:16px 0 6px}
nav a{display:flex;justify-content:space-between;padding:6px 8px;border-radius:6px;color:#ddd;text-decoration:none;cursor:pointer}
nav a.on,nav a:hover{background:#2a2a2a}nav small{color:#888}
main{flex:1;display:flex;flex-direction:column;min-width:0}
header{padding:10px 16px;border-bottom:1px solid #333;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
header h2{margin:0 12px 0 0;font-size:18px}button,select,input{background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 10px;font:inherit}
button:hover{background:#333}button.warn{border-color:#a44}button.ok{border-color:#4a4}
#grid{flex:1;overflow:auto;padding:16px;display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start}
.t{width:112px;cursor:pointer;position:relative}.t img{width:112px;height:112px;object-fit:cover;border-radius:6px;display:block;border:3px solid transparent}
.t.sel img{border-color:#4af}.t.auto img{opacity:.8}.t .cap{font-size:11px;color:#999;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.t .badge{position:absolute;top:4px;left:4px;font-size:10px;background:#a44;color:#fff;padding:1px 4px;border-radius:4px}
.t .auto-b{position:absolute;top:4px;right:4px;font-size:10px;background:#555;padding:1px 4px;border-radius:4px}
aside{width:360px;border-left:1px solid #333;padding:12px;overflow:auto}aside img{width:100%;border-radius:6px}
#help{color:#888;font-size:12px;line-height:1.6}kbd{background:#333;padding:0 4px;border-radius:3px}
</style></head><body>
<nav><h3>Review</h3><div id="rev"></div><h3>People</h3><div id="ppl"></div><h3>Unnamed groups</h3><div id="grp"></div></nav>
<main><header id="bar"></header><div id="grid"></div></main>
<aside><img id="prev"><p id="pcap" style="color:#999;font-size:12px"></p>
<div id="help">Click to select, <kbd>Shift</kbd>-click for a range, <kbd>A</kbd> select all, <kbd>Esc</kbd> clear.<br>
<kbd>R</kbd> remove from this person (it's not them)<br><kbd>T</kbd> toggle "exclude from training" (it's them, but filtered or edited)<br>
<kbd>Y</kbd> in Review: accept the guess<br>Hover a face to see the whole photo here.<br><br>
Faces marked <span style="background:#555;padding:0 4px;border-radius:3px">auto</span> were filed by <code>assign</code>, not by you.</div></aside>
<script>
let S=null,view={kind:'review',name:null},sel=new Set(),last=null;
const $=id=>document.getElementById(id);
async function load(){S=await (await fetch('/api/state')).json();render()}
async function post(u,b){await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});sel.clear();await load()}
function items(){if(view.kind==='review')return S.review;if(view.kind==='group')return S.unnamed_groups[view.name]||[];return S.people[view.name]||[]}
function nav(){
 $('rev').innerHTML=`<a class="${view.kind==='review'?'on':''}" onclick="go('review')">Needs your answer <small>${S.review.length}</small></a>`;
 $('ppl').innerHTML=Object.entries(S.people).map(([n,v])=>`<a class="${view.kind==='person'&&view.name===n?'on':''}" onclick='go("person",${JSON.stringify(n)})'>${esc(n)} <small>${v.length}</small></a>`).join('');
 $('grp').innerHTML=Object.entries(S.unnamed_groups).map(([n,v])=>`<a class="${view.kind==='group'&&view.name===n?'on':''}" onclick='go("group",${JSON.stringify(n)})'>${esc(n)} <small>${v.length}</small></a>`).join('')||'<small>none</small>'}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function opts(){return Object.keys(S.people).map(n=>`<option>${esc(n)}</option>`).join('')}
function bar(){
 const n=sel.size,dl=`<input id="dst" list="pl" placeholder="person name" style="width:160px"><datalist id="pl">${opts()}</datalist>`;
 let h='';
 if(view.kind==='person'){h=`<h2>${esc(view.name)}</h2><span>${n} selected</span>
  <button class="warn" onclick="act('remove')">Not them (R)</button>
  <button onclick="act('train',true)">Exclude from training (T)</button><button onclick="act('train',false)">Include in training</button>
  ${dl}<button onclick="act('move')">Move to</button><button onclick="renameP()">Rename</button>`}
 else if(view.kind==='review'){h=`<h2>Needs your answer</h2><span>${n} selected</span>
  <button class="ok" onclick="act('accept')">Yes, it's the guess (Y)</button><button class="warn" onclick="act('reject')">No (R)</button>
  ${dl}<button onclick="act('answer')">It's</button>`}
 else{h=`<h2>${esc(view.name)}</h2><span>${n} selected</span>${dl}<button class="ok" onclick="act('name')">Name selected (or all if none)</button>`}
 $('bar').innerHTML=h}
function grid(){
 const it=items();$('grid').innerHTML=it.map((t,i)=>`<div class="t ${sel.has(t.id)?'sel':''} ${t.source==='auto'?'auto':''}" data-i="${i}">
 <img src="/crops/${t.crop}" loading="lazy">${t.no_training?'<span class="badge">no train</span>':''}${t.source==='auto'?'<span class="auto-b">auto</span>':''}
 <div class="cap">${t.guess!==undefined?esc((t.guess||'?')+' '+t.score.toFixed(2)+(t.second_pass?' 2nd':'')):esc(t.photo)}</div></div>`).join('')||'<p style="color:#888">Nothing here.</p>';
 document.querySelectorAll('.t').forEach(el=>{const i=+el.dataset.i,t=it[i];
  el.onclick=e=>{if(e.shiftKey&&last!==null){const [a,b]=[Math.min(last,i),Math.max(last,i)];for(let k=a;k<=b;k++)sel.add(it[k].id)}else{sel.has(t.id)?sel.delete(t.id):sel.add(t.id)}last=i;grid();bar()};
  el.onmouseenter=()=>{$('prev').src='/photo?face='+encodeURIComponent(t.id);$('pcap').textContent=t.photo}})}
function render(){nav();bar();grid()}
function go(kind,name=null){view={kind,name};sel.clear();last=null;render()}
async function act(a,flag){
 const f=[...sel];const dst=($('dst')||{}).value;
 if(view.kind==='group'&&a==='name'){if(!dst)return alertish('type a name');const fs=f.length?f:items().map(t=>t.id);return post('/api/move',{faces:fs,dst})}
 if(!f.length)return alertish('select some faces first');
 if(a==='remove')return post('/api/remove',{person:view.name,faces:f});
 if(a==='train')return post('/api/training',{faces:f,exclude:flag});
 if(a==='move'){if(!dst)return alertish('type or pick a person');return post('/api/move',{src:view.name,dst,faces:f})}
 const byId=Object.fromEntries(items().map(t=>[t.id,t]));
 for(const id of f){const name=a==='accept'?byId[id].guess:a==='answer'?dst:null;if(a==='answer'&&!dst)return alertish('type a name');
  await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({face:id,name})})}
 sel.clear();await load()}
async function renameP(){const n=prompt('New name (an existing name merges the two):',view.name);if(n&&n!==view.name){await post('/api/rename',{old:view.name,new:n});go('person',n)}}
function alertish(m){$('pcap').textContent='⚠ '+m}
document.onkeydown=e=>{if(e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();
 if(k==='escape'){sel.clear();grid();bar()}else if(k==='a'){items().forEach(t=>sel.add(t.id));grid();bar()}
 else if(k==='r')act(view.kind==='review'?'reject':'remove');else if(k==='t'&&view.kind==='person'){const it=items().filter(t=>sel.has(t.id));act('train',!it.every(t=>t.no_training))}
 else if(k==='y'&&view.kind==='review')act('accept')};
load().then(()=>{const p=Object.keys(S.people);if(!S.review.length&&p.length)go('person',p[0])});
</script></body></html>"""
