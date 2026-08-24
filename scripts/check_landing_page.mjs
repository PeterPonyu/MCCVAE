#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const landingPath = path.join(root, 'out', 'index.html');
const canonicalUrl = 'https://peterponyu.github.io/MCCVAE/';
const requiredLinks = [
  'https://peterponyu.github.io/',
  'https://peterponyu.github.io/scportal/',
  'https://github.com/PeterPonyu/MCCVAE',
];

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

assert(fs.existsSync(landingPath), 'Landing artifact must contain out/index.html.');
const html = fs.readFileSync(landingPath, 'utf8');

const canonicalMatch = html.match(/<link\b[^>]*\brel=["']canonical["'][^>]*\bhref=["']([^"']+)["'][^>]*>/i);
assert(canonicalMatch?.[1] === canonicalUrl, `Landing canonical must be exactly ${canonicalUrl}.`);

const robotsMatch = html.match(/<meta\b[^>]*\bname=["']robots["'][^>]*\bcontent=["']([^"']+)["'][^>]*>/i);
assert(robotsMatch?.[1].trim().toLowerCase() === 'noindex, follow', 'Landing robots metadata must be exactly "noindex, follow".');

for (const link of requiredLinks) {
  assert(html.includes(`href="${link}"`) || html.includes(`href='${link}'`), `Landing must link to ${link}.`);
}

assert(!/<form\b/i.test(html), 'Landing must not contain operational forms.');
assert(!/<input\b[^>]*\btype=["']?file\b/i.test(html), 'Landing must not contain file-upload controls.');

const urls = [...html.matchAll(/\b(?:href|src|action|content)=["'](https?:\/\/[^"']+)["']/gi)].map((match) => match[1]);
const backendUrl = /:\d+(?:\/|$)|\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0)\b|\/\/(?:api|backend|service)\.|\/(?:api|upload|service)(?:\/|$)/i;
assert(!urls.some((url) => backendUrl.test(url)), 'Landing must not contain backend or service URLs.');

console.log('MCCVAE landing page contract passed.');
