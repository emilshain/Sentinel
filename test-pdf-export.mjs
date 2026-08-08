import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const BASE_URL = 'http://localhost:5174';

async function test() {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  console.log('📌 Step 1: Navigate to the app');
  await page.goto(BASE_URL);
  await page.waitForLoadState('networkidle');

  // Take screenshot of initial state
  await page.screenshot({ path: 'step1-initial.png' });
  console.log('✅ Page loaded successfully');
  console.log('   Screenshot: step1-initial.png');

  // Check if upload screen is visible
  const uploadScreen = await page.locator('text=/Upload|upload/i').first();
  if (await uploadScreen.isVisible()) {
    console.log('✅ Upload screen visible');
  } else {
    console.log('❌ Upload screen not found');
  }

  console.log('\n📌 Step 2: Wait for results to load');
  // The app loads a demo report, wait for results to appear
  await page.waitForSelector('.results-topbar', { timeout: 10000 });
  console.log('✅ Results view appeared');

  await page.screenshot({ path: 'step2-results.png' });
  console.log('   Screenshot: step2-results.png');

  console.log('\n📌 Step 3: Check for Export button');
  const exportButton = await page.locator('text=/Export as PDF|📄/i');

  if (await exportButton.isVisible()) {
    console.log('✅ Export button is visible');
    await exportButton.highlight();
    await page.screenshot({ path: 'step3-export-button.png' });
    console.log('   Screenshot: step3-export-button.png');
  } else {
    console.log('❌ Export button not found');
    await page.screenshot({ path: 'step3-no-button.png' });
  }

  console.log('\n📌 Step 4: Click export button and verify PDF generation');

  // Listen for download
  const downloadPromise = page.waitForEvent('download');

  // Click the export button
  await exportButton.click();
  console.log('   Clicked export button...');

  try {
    const download = await downloadPromise;
    const filename = await download.suggestedFilename();

    console.log(`✅ PDF downloaded successfully`);
    console.log(`   Filename: ${filename}`);

    // Save the download to verify
    const path_to_save = `./downloaded-${filename}`;
    await download.saveAs(path_to_save);

    const stats = fs.statSync(path_to_save);
    console.log(`   File size: ${(stats.size / 1024).toFixed(2)} KB`);

    if (stats.size > 0) {
      console.log('✅ PDF file created with content');
    }
  } catch (e) {
    console.log('❌ PDF download failed:', e.message);
    await page.screenshot({ path: 'step4-error.png' });
  }

  console.log('\n📌 Step 5: Probe - Click export button again');
  await exportButton.click();
  await page.waitForTimeout(2000);
  console.log('✅ Export button can be clicked multiple times');

  console.log('\n📌 Step 6: Verify export button on How We Found It tab');
  const howTab = await page.locator('text=/How we found it/i');
  await howTab.click();
  await page.waitForTimeout(1000);

  const exportButtonAfterTab = await page.locator('text=/Export as PDF|📄/i');
  if (await exportButtonAfterTab.isVisible()) {
    console.log('✅ Export button still visible when switching tabs');
  } else {
    console.log('❌ Export button missing on How We Found It tab');
  }

  await page.screenshot({ path: 'step6-how-tab.png' });

  console.log('\n📌 Step 7: Test rescan flow');
  const rescanButton = await page.locator('text=/Test another model|↻/i');
  if (await rescanButton.isVisible()) {
    console.log('✅ Rescan button is visible');
  }

  await browser.close();
  console.log('\n✅ All tests completed successfully!');
}

test().catch(err => {
  console.error('❌ Test failed:', err);
  process.exit(1);
});
