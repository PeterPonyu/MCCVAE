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

const isWhitespace = (character) => /\s/.test(character);
const isAsciiLetter = (character) => (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z');
const isAttributeDelimiter = (character) => isWhitespace(character) || character === '=' || character === '/' || character === '>';
const localName = (name) => name.split(':').at(-1);

const parseTag = (source) => {
  let index = 1;
  while (isWhitespace(source[index] ?? '')) index += 1;
  if (!isAsciiLetter(source[index] ?? '')) return null;

  const nameStart = index;
  while (/[a-z0-9:-]/i.test(source[index] ?? '')) index += 1;
  const name = source.slice(nameStart, index).toLowerCase();
  const attributes = new Map();

  while (index < source.length - 1) {
    while (isWhitespace(source[index] ?? '')) index += 1;
    if (source[index] === '/') {
      let remainder = index + 1;
      while (isWhitespace(source[remainder] ?? '')) remainder += 1;
      assert(source[remainder] === '>', 'Landing start tag has an ambiguous slash before attributes.');
      break;
    }
    if (source[index] === '>') break;

    const attributeStart = index;
    while (!isAttributeDelimiter(source[index] ?? '')) index += 1;
    const attributeName = source.slice(attributeStart, index).toLowerCase();
    if (!attributeName) return null;

    while (isWhitespace(source[index] ?? '')) index += 1;
    let value = '';
    if (source[index] === '=') {
      index += 1;
      while (isWhitespace(source[index] ?? '')) index += 1;
      const quote = source[index];
      if (quote === '"' || quote === "'") {
        index += 1;
        const valueStart = index;
        while (index < source.length && source[index] !== quote) index += 1;
        if (source[index] !== quote) return null;
        value = source.slice(valueStart, index);
        index += 1;
      } else {
        const valueStart = index;
        while (index < source.length && !isWhitespace(source[index] ?? '') && source[index] !== '>') index += 1;
        value = source.slice(valueStart, index);
      }
    }
    const values = attributes.get(attributeName) ?? [];
    values.push(value);
    attributes.set(attributeName, values);
  }

  return { name, attributes };
};

const tokenizeTags = (html) => {
  const matches = [];
  for (let start = html.indexOf('<'); start >= 0; start = html.indexOf('<', start + 1)) {
    let nameIndex = start + 1;
    while (isWhitespace(html[nameIndex] ?? '')) nameIndex += 1;
    const startTagCandidate = isAsciiLetter(html[nameIndex] ?? '');
    let quote = null;
    let closed = false;
    for (let end = start + 1; end < html.length; end += 1) {
      const character = html[end];
      if (quote) {
        if (character === quote) quote = null;
      } else if (character === '"' || character === "'") {
        quote = character;
      } else if (character === '>') {
        closed = true;
        if (startTagCandidate) {
          const tag = parseTag(html.slice(start, end + 1));
          assert(tag, 'Landing contains a malformed start tag.');
          matches.push(tag);
        }
        start = end;
        break;
      }
    }
    assert(!startTagCandidate || closed, 'Landing contains an unterminated start tag.');
  }
  return matches;
};

const tags = (html, name) => tokenizeTags(html).filter((tag) => tag.name === name.toLowerCase());

const sensitiveAttributeValues = (tag, name) => {
  const values = [...tag.attributes].flatMap(([attributeName, attributeValues]) => (localName(attributeName) === name ? attributeValues : []));
  assert(values.length <= 1, `Landing tag has duplicate security-sensitive ${name} attributes.`);
  return values;
};

const securityAttributeEntries = (html) =>
  tokenizeTags(html).flatMap((tag) =>
    ['href', 'src', 'action', 'type'].flatMap((name) =>
      sensitiveAttributeValues(tag, name).map((value) => ({ tag, name, value })),
    ),
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

const decodeSecurityValue = (value) => {
  let decoded = value;
  for (let pass = 0; pass < 4; pass += 1) {
    const prior = decoded;
    decoded = decoded.replace(/&#(?:x([0-9a-f]+)|([0-9]+));?/gi, (reference, hex, decimal) => {
      if (hex || decimal) {
        const codePoint = Number.parseInt(hex ?? decimal, hex ? 16 : 10);
        assert(Number.isInteger(codePoint) && codePoint >= 0 && codePoint <= 0x10ffff, `Landing URL has an invalid HTML entity: ${reference}.`);
        return String.fromCodePoint(codePoint);
      }
      return reference;
    });
    decoded = decoded.replace(/&([a-z][a-z0-9]+);/gi, (reference, named) => {
      const replacement = namedEntities.get(named.toLowerCase());
      assert(replacement !== undefined, `Landing URL has an unsupported or ambiguous HTML entity: ${reference}.`);
      return replacement;
    });
    if (decoded === prior) break;
  }
  assert(!/&#(?:x[0-9a-f]*|[0-9]*)/i.test(decoded), `Landing URL has an unsupported or ambiguous HTML entity: ${value}.`);
  assert(!/&[a-z][a-z0-9]*;?/i.test(decoded), `Landing URL has an unsupported or ambiguous HTML entity: ${value}.`);
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
  const canonical = tags(html, 'link').find((tag) => tag.attributes.get('rel')?.length === 1 && tag.attributes.get('rel')[0].toLowerCase() === 'canonical');
  assert(
    canonical && decodeSecurityValue(sensitiveAttributeValues(canonical, 'href')[0] ?? '') === canonicalUrl,
    `Landing canonical must be exactly ${canonicalUrl}.`,
  );

  const robots = tags(html, 'meta').find((tag) => tag.attributes.get('name')?.length === 1 && tag.attributes.get('name')[0].toLowerCase() === 'robots');
  assert(robots?.attributes.get('content')?.length === 1 && robots.attributes.get('content')[0].trim().toLowerCase() === 'noindex, follow', 'Landing robots metadata must be exactly "noindex, follow".');

  const securityAttributes = securityAttributeEntries(html);
  const urlAttributes = securityAttributes.filter(({ name }) => ['href', 'src', 'action'].includes(name));
  const hrefs = urlAttributes.filter(({ name }) => name === 'href').map(({ value }) => decodeSecurityValue(value));
  for (const link of requiredLinks) {
    assert(hrefs.includes(link), `Landing must link to ${link}.`);
  }

  assert(!tags(html, 'form').length, 'Landing must not contain operational forms.');
  assert(
    !tags(html, 'input').some((tag) => decodeSecurityValue(sensitiveAttributeValues(tag, 'type')[0] ?? '').toLowerCase() === 'file'),
    'Landing must not contain file-upload controls.',
  );

  for (const { name, value } of urlAttributes) {
    const decodedValue = decodeSecurityValue(value);
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

const expectPass = (name, html) => {
  validateLandingHtml(html);
  console.log(`PASS ${name}`);
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
    ['fake-href-in-data-note', '<a data-note=\'href="https://peterponyu.github.io/"\' href="/api/status">API</a>', 'operational path'],
    ['fake-type-in-data-note', '<input data-note=\'type="text"\' type="file">', 'file-upload controls'],
    ['entity-encoded-file-type', '<input type="f&#105;le">', 'file-upload controls'],
    ['entity-encoded-api-path', '<a href="&#47;api/status">API</a>', 'operational path'],
    ['semicolonless-decimal-api-path', '<a href="&#47api/status">API</a>', 'operational path'],
    ['semicolonless-hex-api-path', '<a href="&#x2f/api/status">API</a>', 'backend or loopback host'],
    ['namespaced-href-api-path', '<svg xlink:href="/api/status"></svg>', 'operational path'],
    ['duplicate-href', '<a href="https://peterponyu.github.io/" href="/api/status">API</a>', 'duplicate security-sensitive href'],
    ['double-encoded-api-path', '<a href="&amp;#47;api/status">API</a>', 'operational path'],
    ['slash-before-href-with-space', '<a / href="/api/status">API</a>', 'ambiguous slash before attributes'],
    ['slash-before-href-without-space', '<a/href="/api/status">API</a>', 'ambiguous slash before attributes'],
    ['slash-after-partial-attribute-href', '<a x/href="/api/status">API</a>', 'ambiguous slash before attributes'],
    ['slash-after-partial-attribute-type', '<input x/type="file">', 'ambiguous slash before attributes'],
    ['unterminated-start-tag', '<a href="/api/status', 'unterminated start tag'],
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
  expectPass('self-closing-tag', validHtml.replace('</body>', '<img src="/assets/icon.svg" /></body>'));
  expectPass('url-path-with-slashes', validHtml.replace('</body>', '<a href="/MCCVAE/docs/guide">Guide</a></body>'));
  expectPass('quoted-custom-attribute-slash', validHtml.replace('</body>', '<div data-note="safe/value"></div></body>'));
  console.log('MCCVAE landing page self-test passed.');
};

if (process.argv.includes('--self-test')) {
  runSelfTest();
} else {
  assert(fs.existsSync(landingPath), 'Landing artifact must contain out/index.html.');
  validateLandingHtml(fs.readFileSync(landingPath, 'utf8'));
  console.log('MCCVAE landing page contract passed.');
}
