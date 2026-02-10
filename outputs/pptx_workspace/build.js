const pptxgen = require('pptxgenjs');
const html2pptx = require('/Users/alessandrodibari/.claude/skills/pptx/scripts/html2pptx');
const path = require('path');

async function build() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Alessandro Di Bari';
  pptx.title = 'Procedural Posture Classification - Feasibility Study';

  const slidesDir = path.join(__dirname, 'slides');
  const slides = [
    'slide01_title.html',
    'slide02_what_is_posture.html',
    'slide03_dataset.html',
    'slide04_label_dist.html',
    'slide05_domain.html',
    'slide06_section_solution.html',
    'slide07_strategy.html',
    'slide08_baseline.html',
    'slide09_long_doc.html',
    'slide10_architecture.html',
    'slide11_section_repo.html',
    'slide12_project_org.html',
    'slide13_design.html',
    'slide14_section_results.html',
    'slide15_baseline_results.html',
    'slide16_hybrid_results.html',
    'slide17_feasibility.html',
    'slide18_insights.html',
    'slide19_next_steps.html',
    'slide20_thankyou.html',
  ];

  for (let i = 0; i < slides.length; i++) {
    const file = path.join(slidesDir, slides[i]);
    console.log(`Processing ${slides[i]}...`);
    await html2pptx(file, pptx);
  }

  const outputPath = path.join(__dirname, '..', 'TR_Interview_Presentation.pptx');
  await pptx.writeFile({ fileName: outputPath });
  console.log(`\nPresentation saved to: ${outputPath}`);
}

build().catch(err => { console.error(err); process.exit(1); });
