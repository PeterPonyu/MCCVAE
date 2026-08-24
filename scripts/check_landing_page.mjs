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

const tagName = (tag) => tag.match(/^<\s*([a-z][a-z0-9:-]*)\b/i)?.[1].toLowerCase() ?? null;

const tokenizeTags = (html) => {
  const matches = [];
  for (let start = html.indexOf('<'); start >= 0; start = html.indexOf('<', start + 1)) {
    let quote = null;
    for (let end = start + 1; end < html.length; end += 1) {
      const character = html[end];
      if (quote) {
        if (character === quote) quote = null;
      } else if (character === '"' || character === "'") {
        quote = character;
      } else if (character === '>') {
        const tag = html.slice(start, end + 1);
        if (tagName(tag)) matches.push(tag);
        start = end;
        break;
      }
    }
  }
  return matches;
};

const tags = (html, name) => tokenizeTags(html).filter((tag) => tagName(tag) === name.toLowerCase());

const attributeValue = (tag, name) => {
  const match = tag.match(
    new RegExp(`(?<![\\w:-])${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s"'=<>\\x60]+))`, 'i'),
  );
  return match?.[1] ?? match?.[2] ?? match?.[3] ?? null;
};

const attributeValues = (html, names) =>
  tokenizeTags(html).flatMap((tag) =>
    names.flatMap((name) => {
      const value = attributeValue(tag, name);
      return value === null ? [] : [{ name, value }];
    }),
  );

const namedEntities = new Map([
  ['amp', '&'],
  ['apos', "'"],
  ['colon', ':'],
  ['gt', '>'],
  ['lt', '<'],
  ['period', '.'],
  ['quot', '"'],
  ['sol', '/'],
]);

const decodeSecurityUrl = (value) => {
  let decoded = value;
  for (let pass = 0; pass < 8; pass += 1) {
    if (!/&(?:#x[0-9a-f]+|#[0-9]+|[a-z][a-z0-9]+);/i.test(decoded)) break;
    decoded = decoded.replace(/&(?:#x([0-9a-f]+)|#([0-9]+)|([a-z][a-z0-9]+));/gi, (reference, hex, decimal, named) => {
      if (hex || decimal) {
        const codePoint = Number.parseInt(hex ?? decimal, hex ? 16 : 10);
        assert(Number.isInteger(codePoint) && codePoint >= 0 && codePoint <= 0x10ffff, `Landing URL has an invalid HTML entity: ${reference}.`);
        return String.fromCodePoint(codePoint);
      }
      const replacement = namedEntities.get(named.toLowerCase());
      assert(replacement !== undefined, `Landing URL has an unsupported or ambiguous HTML entity: ${reference}.`);
      return replacement;
    });
  }
  assert(!/&(?:#x[0-9a-f]+|#[0-9]+|[a-z][a-z0-9]+);/i.test(decoded), `Landing URL has an unsupported or ambiguous HTML entity: ${value}.`);
  return decoded;
};

const isForbiddenHost = (hostname) => {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '');
  const ipv4 = host.match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)$/)?.slice(1).map(Number);
  const privateIpv4 =
    ipv4 &&
    ipv4.every((segment) => Number.isInteger(segment) && segment >= 0 && segment <= 255) &&
    (ipv4[0] === 0 ||
      ipv4[0] === 10 ||
      ipv4[0] === 127 ||
      (ipv4[0] === 169 && ipv4[1] === 254) ||
      (ipv4[0] === 172 && ipv4[1] >= 16 && ipv4[1] <= 31) ||
      (ipv4[0] === 192 && ipv4[1] === 168));
  return (
    host === 'localhost' ||
    host.endsWith('.localhost') ||
    host === '::1' ||
    host === '::' ||
    host.startsWith('::ffff:') ||
    /^f[cd][0-9a-f:]*$/i.test(host) ||
    /^fe[89ab][0-9a-f:]*$/i.test(host) ||
    privateIpv4 ||
    /(^|\.)(?:api|backend|service)(?:\.|$)/.test(host)
  );
};

const hasForbiddenPath = (url) =>
  url.pathname
    .split('/')
    .filter(Boolean)
    .map((segment) => decodeURIComponent(segment).toLowerCase())
    .some((segment) => ['api', 'upload', 'service'].includes(segment));

const hasExplicitPort = (value, url) => url.port !== '' || /^https?:\/\/[^/?#]+:\d+(?:[/?#]|$)/i.test(value);

export function validateLandingHtml(html) {
  const canonical = tags(html, 'link').find((tag) => attributeValue(tag, 'rel')?.toLowerCase() === 'canonical');
  assert(attributeValue(canonical ?? '', 'href') === canonicalUrl, `Landing canonical must be exactly ${canonicalUrl}.`);

  const robots = tags(html, 'meta').find((tag) => attributeValue(tag, 'name')?.toLowerCase() === 'robots');
  assert(attributeValue(robots ?? '', 'content')?.trim().toLowerCase() === 'noindex, follow', 'Landing robots metadata must be exactly "noindex, follow".');

  const urlAttributes = attributeValues(html, ['href', 'src', 'action']);
  const hrefs = urlAttributes.filter(({ name }) => name === 'href').map(({ value }) => value);
  for (const link of requiredLinks) {
    assert(hrefs.includes(link), `Landing must link to ${link}.`);
  }

  assert(!/<form\b/i.test(html), 'Landing must not contain operational forms.');
  assert(
    !tags(html, 'input').some((tag) => attributeValue(tag, 'type')?.toLowerCase() === 'file'),
    'Landing must not contain file-upload controls.',
  );

  for (const { name, value } of urlAttributes) {
    const decodedValue = decodeSecurityUrl(value);
    const url = new URL(decodedValue, canonicalUrl);
    if (!['http:', 'https:'].includes(url.protocol)) continue;
    assert(!isForbiddenHost(url.hostname), `Landing ${name} must not target a backend or loopback host: ${value}.`);
    assert(!hasExplicitPort(decodedValue, url), `Landing ${name} must not target an explicit service port: ${value}.`);
    assert(!hasForbiddenPath(url), `Landing ${name} must not target an operational path: ${value}.`);
  }
}

const expectFailure = (name, html, expectedMessage) => {
  try {
    validateLandingHtml(html);
  } catch (error) {
    assert(error instanceof Error && error.message.includes(expectedMessage), `${name} failed for the wrong reason: ${error}`);
    console.log(`PASS ${name}: ${expectedMessage}`);
    return;
  }
  throw new Error(`${name} unexpectedly passed.`);
};

const runSelfTest = () => {
  assert(fs.existsSync(landingPath), 'Landing artifact must contain out/index.html.');
  const validHtml = fs.readFileSync(landingPath, 'utf8');
  validateLandingHtml(validHtml);
  console.log('PASS valid-tracked-artifact');

  const cases = [
    ['relative-api-path', '<a href = "/api/status">API</a>', 'operational path'],
    ['whitespace-file-control', '<input type = "file">', 'file-upload controls'],
    ['quoted-gt-before-api-path', '<a data-note=">" href="/api/status">API</a>', 'operational path'],
    ['quoted-gt-before-file-control', '<input data-note=">" type="file">', 'file-upload controls'],
    ['entity-encoded-api-path', '<a href="&#47;api/status">API</a>', 'operational path'],
    ['root-relative-upload-path', '<img src="/upload/model">', 'operational path'],
    ['root-relative-service-path', '<button action = "/service/run">Service</button>', 'operational path'],
    ['localhost-endpoint', '<a href="http://localhost:3000/status">Local</a>', 'backend or loopback host'],
    ['ipv6-loopback-endpoint', '<a href="http://[::1]/status">IPv6 loopback</a>', 'backend or loopback host'],
    ['ipv4-mapped-loopback-endpoint', '<a href="http://[::ffff:127.0.0.1]/status">Mapped loopback</a>', 'backend or loopback host'],
    ['backend-host', '<a href="https://backend.example.com/status">Backend</a>', 'backend or loopback host'],
    ['explicit-port-endpoint', '<a href="https://example.com:8443/status">Port</a>', 'explicit service port'],
  ];
  for (const [name, injection, expectedMessage] of cases) {
    expectFailure(name, validHtml.replace('</body>', `${injection}</body>`), expectedMessage);
  }
  console.log('MCCVAE landing page self-test passed.');
};

if (process.argv.includes('--self-test')) {
  runSelfTest();
} else {
  assert(fs.existsSync(landingPath), 'Landing artifact must contain out/index.html.');
  validateLandingHtml(fs.readFileSync(landingPath, 'utf8'));
  console.log('MCCVAE landing page contract passed.');
}
