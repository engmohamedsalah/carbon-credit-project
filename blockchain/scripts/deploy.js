const hre = require("hardhat");

async function main() {
  console.log("Deploying CarbonCreditNFT contract...");
  
  const CarbonCreditNFT = await hre.ethers.getContractFactory("CarbonCreditNFT");
  const carbonCreditNFT = await CarbonCreditNFT.deploy();
  
  await carbonCreditNFT.waitForDeployment();
  
  const contractAddress = await carbonCreditNFT.getAddress();
  console.log("CarbonCreditNFT deployed to:", contractAddress);

  // Save deployment info with the full JSON ABI (array of objects) that web3 needs.
  const fs = require('fs');
  const artifact = await hre.artifacts.readArtifact("CarbonCreditNFT");
  const deploymentInfo = {
    contractAddress: contractAddress,
    network: hre.network.name,
    deploymentTime: new Date().toISOString(),
    abi: artifact.abi
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