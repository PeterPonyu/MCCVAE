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

const tags = (html, name) => html.match(new RegExp(`<${name}\\b[^>]*>`, 'gi')) ?? [];

const attributeValue = (tag, name) => {
  const match = tag.match(
    new RegExp(`(?<![\\w:-])${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s"'=<>\\x60]+))`, 'i'),
  );
  return match?.[1] ?? match?.[2] ?? match?.[3] ?? null;
};

const attributeValues = (html, names) =>
  tags(html, '[a-z][a-z0-9:-]*').flatMap((tag) =>
    names.flatMap((name) => {
      const value = attributeValue(tag, name);
      return value === null ? [] : [{ name, value }];
    }),
  );

const isForbiddenHost = (hostname) => {
  const host = hostname.toLowerCase();
  return (
    host === 'localhost' ||
    host.endsWith('.localhost') ||
    host === '0.0.0.0' ||
    host === '::1' ||
    host.startsWith('127.') ||
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
    const url = new URL(value, canonicalUrl);
    if (!['http:', 'https:'].includes(url.protocol)) continue;
    assert(!isForbiddenHost(url.hostname), `Landing ${name} must not target a backend or loopback host: ${value}.`);
    assert(!hasExplicitPort(value, url), `Landing ${name} must not target an explicit service port: ${value}.`);
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
    ['root-relative-upload-path', '<img src="/upload/model">', 'operational path'],
    ['root-relative-service-path', '<button action = "/service/run">Service</button>', 'operational path'],
    ['localhost-endpoint', '<a href="http://localhost:3000/status">Local</a>', 'backend or loopback host'],
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
