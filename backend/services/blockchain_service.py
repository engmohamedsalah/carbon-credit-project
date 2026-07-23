"""
Blockchain Service for Carbon Credit NFT Management
Simple implementation using Web3.py and Polygon network
"""
import json
import os
from typing import Optional, Dict, Any
import logging

# web3/eth_account are optional: if absent, the feature is cleanly disabled
# (endpoints return 503) rather than crashing the app on import.
try:
    from web3 import Web3
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional deps
    Web3 = None
    Account = None
    WEB3_AVAILABLE = False

logger = logging.getLogger(__name__)

class BlockchainService:
    def __init__(self):
        # Default RPC targets the Polygon Amoy testnet (Mumbai was deprecated in 2024).
        self.rpc_url = os.getenv("POLYGON_RPC_URL", "https://rpc-amoy.polygon.technology")
        self.private_key = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
        self.contract_address = os.getenv("CONTRACT_ADDRESS")

        self.w3 = None
        self.account = None
        self.contract = None
        self.contract_abi = self._load_contract_abi() if WEB3_AVAILABLE else None

        if WEB3_AVAILABLE:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            if self.private_key:
                self.account = Account.from_key(self.private_key)
            if self.contract_address and self.contract_abi:
                self.contract = self.w3.eth.contract(
                    address=self.contract_address,
                    abi=json.loads(self.contract_abi)
                )

        # Feature is enabled only when fully configured for real on-chain operations.
        self.enabled = bool(WEB3_AVAILABLE and self.contract and self.account)
        if not self.enabled:
            logger.info(
                "Blockchain certification disabled (web3=%s, contract=%s, account=%s) "
                "— endpoints will report it as not configured.",
                WEB3_AVAILABLE, bool(self.contract), bool(self.account),
            )
    
    def _load_contract_abi(self) -> Optional[str]:
        """Load contract ABI from blockchain_config.json"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "..", "blockchain_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('abi')
        except Exception as e:
            logger.warning(f"Could not load contract ABI: {e}")
        return None
    
    def is_connected(self) -> bool:
        """Check if connected to blockchain network"""
        try:
            return bool(self.w3 and self.w3.is_connected())
        except Exception:
            return False
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get basic network information"""
        try:
            if not self.is_connected():
                return {"error": "Not connected to network"}
            
            chain_id = self.w3.eth.chain_id
            latest_block = self.w3.eth.get_block('latest')
            
            return {
                "connected": True,
                "chain_id": chain_id,
                "network": "Polygon Amoy" if chain_id == 80002 else "Polygon Mainnet" if chain_id == 137 else f"Chain {chain_id}",
                "latest_block": latest_block.number,
                "gas_price": self.w3.eth.gas_price
            }
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return {"error": str(e)}
    
    def mint_carbon_credit_nft(self, recipient_address: str, project_id: int, 
                              carbon_amount: int, project_name: str, 
                              location: str, verification_hash: str) -> Optional[Dict[str, Any]]:
        """Mint a carbon credit NFT"""
        try:
            if not self.contract or not self.account:
                return {"error": "Contract or account not initialized"}
            
            # Create token URI (simplified - in production would be IPFS)
            token_uri = f"https://api.carbonverify.com/metadata/{project_id}"
            
            # Build transaction
            tx_function = self.contract.functions.mintCarbonCredit(
                recipient_address,
                project_id,
                carbon_amount,
                project_name,
                location,
                verification_hash,
                token_uri
            )
            
            # Get gas estimate
            gas_estimate = tx_function.estimate_gas({'from': self.account.address})
            
            # Build transaction
            transaction = tx_function.build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': int(gas_estimate * 1.2),  # Add 20% buffer
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign transaction
            signed_txn = self.account.sign_transaction(transaction)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            # Get token ID from logs
            token_id = None
            for log in tx_receipt.logs:
                try:
                    decoded_log = self.contract.events.CarbonCreditMinted().process_log(log)
                    token_id = decoded_log['args']['tokenId']
                    break
                except:
                    continue
            
            return {
                "success": True,
                "transaction_hash": tx_hash.hex(),
                "token_id": token_id,
                "gas_used": tx_receipt.gasUsed,
                "block_number": tx_receipt.blockNumber
            }
            
        except Exception as e:
            logger.error(f"Error minting NFT: {e}")
            return {"error": str(e)}
    
    def get_carbon_credit_info(self, token_id: int) -> Optional[Dict[str, Any]]:
        """Get carbon credit information by token ID"""
        try:
            if not self.contract:
                return {"error": "Contract not initialized"}
            
            # Call contract function
            carbon_credit = self.contract.functions.getCarbonCredit(token_id).call()
            
            return {
                "token_id": token_id,
                "project_id": carbon_credit[0],
                "carbon_amount": carbon_credit[1],
                "project_name": carbon_credit[2],
                "location": carbon_credit[3],
                "verification_date": carbon_credit[4],
                "verification_hash": carbon_credit[5],
                "is_retired": carbon_credit[6]
            }
            
        except Exception as e:
            logger.error(f"Error getting carbon credit info: {e}")
            return {"error": str(e)}
    
    def verify_certificate(self, token_id_or_hash: str) -> Dict[str, Any]:
        """Verify a carbon credit certificate by token ID or transaction hash.

        Reads real on-chain state; returns an explicit error when the feature is
        not configured. No demo/mock certificates are ever returned.
        """
        try:
            if not self.enabled:
                return {"error": "Blockchain certification is not configured"}

            # Try to parse as token ID first
            try:
                token_id = int(token_id_or_hash)
                return self.get_carbon_credit_info(token_id) or {"error": "Token not found"}
            except ValueError:
                pass

            # Try as transaction hash
            if token_id_or_hash.startswith('0x'):
                if self.w3 and self.w3.is_connected():
                    try:
                        tx_receipt = self.w3.eth.get_transaction_receipt(token_id_or_hash)
                        # Look for minting event in receipt
                        for log in tx_receipt.logs:
                            try:
                                if self.contract:
                                    decoded_log = self.contract.events.CarbonCreditMinted().process_log(log)
                                    token_id = decoded_log['args']['tokenId']
                                    return self.get_carbon_credit_info(token_id)
                            except:
                                continue
                        return {"error": "No carbon credit found in transaction"}
                    except:
                        return {"error": "Transaction not found"}
                else:
                    return {"error": "Not connected to blockchain network"}
            
            return {"error": "Invalid token ID or transaction hash"}
            
        except Exception as e:
            logger.error(f"Error verifying certificate: {e}")
            return {"error": str(e)}

# Global instance
blockchain_service = BlockchainService()