const hre = require("hardhat");

async function main() {
  console.log("Deploying CarbonCreditNFT contract...");
  
  const CarbonCreditNFT = await hre.ethers.getContractFactory("CarbonCreditNFT");
  const carbonCreditNFT = await CarbonCreditNFT.deploy();
  
  await carbonCreditNFT.waitForDeployment();
  
  const contractAddress = await carbonCreditNFT.getAddress();
  console.log("CarbonCreditNFT deployed to:", contractAddress);
  
  // Save deployment info
  const fs = require('fs');
  const deploymentInfo = {
    contractAddress: contractAddress,
    network: hre.network.name,
    deploymentTime: new Date().toISOString(),
    abi: carbonCreditNFT.interface.format('json')
  };
  
  fs.writeFileSync(
    `../backend/blockchain_config.json`,
    JSON.stringify(deploymentInfo, null, 2)
  );
  
  console.log("Deployment info saved to backend/blockchain_config.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});