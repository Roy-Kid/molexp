"""Integration tests for molpy.Compute classes in molexp workflows.

This module tests that real molpy.Compute classes (MCD, PMSD) work seamlessly
as molexp workflow tasks.
"""

import pytest
import numpy as np
from typing import Any

from molexp.workflow import TaskProtocol, register_task


# Skip if molpy is not available
pytest.importorskip("molpy")

from molpy.compute import MCDCompute, PMSDCompute
from molpy.core import Frame
from molpy.core.trajectory import Trajectory
from molpy.core.box import Box




class TestMCDComputeIntegration:
    """Test MCDCompute as a workflow task."""
    
    def test_mcd_is_protocol_compatible(self):
        """Test that MCDCompute implements TaskProtocol."""
        mcd = MCDCompute(tags=["1"], max_dt=1.0, dt=0.1)
        assert isinstance(mcd, TaskProtocol)
    
    def test_mcd_has_required_methods(self):
        """Test that MCDCompute has execute() and dump() methods."""
        mcd = MCDCompute(tags=["1"], max_dt=1.0, dt=0.1)
        assert hasattr(mcd, "execute")
        assert hasattr(mcd, "dump")
        assert callable(mcd.execute)
        assert callable(mcd.dump)
    
    def test_mcd_dump_with_com(self):
        """Test serializing MCDCompute config with center_of_mass."""
        com = {1: 1.008, 6: 12.011}
        mcd = MCDCompute(tags=["1"], max_dt=10.0, dt=0.1, center_of_mass=com)
        config = mcd.dump()
        
        assert config["center_of_mass"] == com
    
    def test_mcd_registration(self):
        """Test registering MCDCompute is rejected (non-Pydantic config)."""
        with pytest.raises(ValueError, match="config_type"):
            register_task(MCDCompute)


class TestPMSDComputeIntegration:
    """Test PMSDCompute as a workflow task."""
    
    def test_pmsd_is_protocol_compatible(self):
        """Test that PMSDCompute implements TaskProtocol."""
        pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=1.0, dt=0.1)
        assert isinstance(pmsd, TaskProtocol)
    
    def test_pmsd_has_required_methods(self):
        """Test that PMSDCompute has execute() and dump() methods."""
        pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=1.0, dt=0.1)
        assert hasattr(pmsd, "execute")
        assert hasattr(pmsd, "dump")
        assert callable(pmsd.execute)
        assert callable(pmsd.dump)
    
    def test_pmsd_dump(self):
        """Test serializing PMSDCompute config."""
        pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=30.0, dt=0.01)
        config = pmsd.dump()
        
        assert config == {
            "cation_type": 1,
            "anion_type": 2,
            "max_dt": 30.0,
            "dt": 0.01,
        }
    
    def test_pmsd_registration(self):
        """Test registering PMSDCompute is rejected (non-Pydantic config)."""
        with pytest.raises(ValueError, match="config_type"):
            register_task(PMSDCompute)


class TestComputeInWorkflow:
    """Test using Compute classes in actual workflows."""
    
    def test_compute_error_handling(self):
        """Test error handling when input is missing."""
        mcd = MCDCompute(tags=["1"], max_dt=1.0, dt=0.1)
        
        with pytest.raises(ValueError, match="Missing required input: input"):
            mcd.execute(wrong_key="value")


class TestComputeConfigRoundtrip:
    """Test config serialization and reconstruction."""
    
    def test_mcd_config_roundtrip(self):
        """Test that MCDCompute config can be dumped and reconstructed."""
        # Create original
        mcd1 = MCDCompute(tags=["1", "2"], max_dt=30.0, dt=0.01)
        
        # Dump config
        config = mcd1.dump()
        
        # Reconstruct
        mcd2 = MCDCompute(**config)
        
        # Verify they have same config
        assert mcd2.dump() == config
        assert mcd2.tags == mcd1.tags
        assert mcd2.max_dt == mcd1.max_dt
        assert mcd2.dt == mcd1.dt
    
    def test_pmsd_config_roundtrip(self):
        """Test that PMSDCompute config can be dumped and reconstructed."""
        # Create original
        pmsd1 = PMSDCompute(cation_type=1, anion_type=2, max_dt=20.0, dt=0.05)
        
        # Dump config
        config = pmsd1.dump()
        
        # Reconstruct
        pmsd2 = PMSDCompute(**config)
        
        # Verify they have same config
        assert pmsd2.dump() == config
        assert pmsd2.cation_type == pmsd1.cation_type
        assert pmsd2.anion_type == pmsd1.anion_type
        assert pmsd2.max_dt == pmsd1.max_dt
        assert pmsd2.dt == pmsd1.dt
