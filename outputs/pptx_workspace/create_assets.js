const sharp = require('sharp');
const path = require('path');

const dir = path.join(__dirname, 'slides');

async function createGradient(filename, color1, color2, angle = '135') {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="810">
    <defs>
      <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:${color1}"/>
        <stop offset="100%" style="stop-color:${color2}"/>
      </linearGradient>
    </defs>
    <rect width="100%" height="100%" fill="url(#g)"/>
  </svg>`;
  await sharp(Buffer.from(svg)).png().toFile(path.join(dir, filename));
}

async function createAccentBar(filename, color, w, h) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">
    <rect width="100%" height="100%" fill="${color}"/>
  </svg>`;
  await sharp(Buffer.from(svg)).png().toFile(path.join(dir, filename));
}

(async () => {
  // Title slide gradient - deep navy to dark teal
  await createGradient('bg_title.png', '#0A1628', '#0F2B3C');
  // Section header gradient
  await createGradient('bg_section.png', '#0D1F2D', '#162D3E');
  // Content slide bg - very dark
  await createGradient('bg_content.png', '#0E1A2B', '#121E30');
  // Final slide gradient
  await createGradient('bg_final.png', '#0A1628', '#0F2B3C');
  // Accent bar (teal)
  await createAccentBar('accent_teal.png', '#00B4D8', 720, 6);
  // Accent bar (orange)
  await createAccentBar('accent_orange.png', '#E76F51', 720, 6);
  console.log('Assets created.');
})();
