// SVG -> 3840px PNG with a local Chromium; no network requests.
// Usage: node export_png.cjs /path/to/puppeteer /path/to/chrome
const fs = require('fs');
const path = require('path');
const puppeteer = require(process.argv[2] || 'puppeteer');
// SVG sources and the check record are versioned under docs/; PNGs are derived
// artefacts and stay in the ignored data/ tree.
const src = path.resolve(__dirname, '../assets');
const out = path.resolve(__dirname, '../../../../data/presentations/seed_refit');
fs.mkdirSync(out, {recursive: true});
(async () => {
  const browser = await puppeteer.launch({executablePath:process.argv[3],headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  const page = await browser.newPage();
  const report=[];
  for (const file of fs.readdirSync(src).filter(n=>n.endsWith('.svg'))) {
    const svg=fs.readFileSync(path.join(src,file),'utf8');
    await page.setContent('<html><head><meta charset="utf-8"></head><body style="margin:0;background:white">'+svg+'</body></html>');
    const size=await page.evaluate(()=>{const s=document.querySelector('svg');const b=s.viewBox.baseVal;return {w:b.width||Number(s.getAttribute('width')),h:b.height||Number(s.getAttribute('height'))};});
    await page.setViewport({width:3840,height:Math.ceil(3840*size.h/size.w),deviceScaleFactor:1});
    await page.evaluate(()=>{const s=document.querySelector('svg');s.style.maxWidth='none';s.style.width='3840px';s.style.height='auto';s.setAttribute('width','3840');s.removeAttribute('height');});
    await page.evaluate(()=>document.fonts.ready);
    const overflow=await page.evaluate(()=>{const s=document.querySelector('svg'),frame=s.getBoundingClientRect();return [...s.querySelectorAll('text')].map(t=>({text:t.textContent,b:t.getBoundingClientRect()})).filter(({b})=>b.left<frame.left-1||b.top<frame.top-1||b.right>frame.right+1||b.bottom>frame.bottom+1).map(({text})=>text);});
    await page.screenshot({path:path.join(out,file.replace(/\.svg$/,'.png')),fullPage:true});
    report.push({file,viewBox:size,width_px:3840,text_outside_canvas:overflow});
  }
  fs.writeFileSync(path.join(src,'render_checks.json'),JSON.stringify(report,null,2));
  await browser.close();
  console.log('Rendered',report.length,'SVGs; text boundary violations:',report.reduce((n,r)=>n+r.text_outside_canvas.length,0));
})().catch(e=>{console.error(e);process.exit(1);});
