// crates/polars-metal-buffer/src/device.rs
//
// Need to link CoreGraphics to pull in MTLCreateSystemDefaultDevice.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use crate::BufferError;

/// A handle to an `MTLDevice`. Constructed once per process; cloning is cheap
/// (it bumps an Objective-C refcount).
#[derive(Clone)]
pub struct MetalDevice {
    inner: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalDevice {
    /// Acquire the system-default Metal device.
    ///
    /// Returns `Err(BufferError::AllocationFailed { bytes: 0 })` when no
    /// Metal-capable GPU is present (e.g. running on a non-Apple-Silicon
    /// machine or inside a CI sandbox without GPU access).
    pub fn system_default() -> Result<Self, BufferError> {
        // SAFETY: MTLCreateSystemDefaultDevice is safe to call; it returns NULL
        // when no device is available (e.g., non-Metal hardware).
        let raw = unsafe { MTLCreateSystemDefaultDevice() };
        // SAFETY: If non-null, the pointer is a valid +1 Objective-C object
        // returned by a function whose name begins with "Create" (CF ownership
        // convention), which objc2's Retained takes ownership of.
        let inner =
            unsafe { Retained::from_raw(raw) }.ok_or(BufferError::AllocationFailed { bytes: 0 })?;
        Ok(Self { inner })
    }

    /// Human-readable name of the GPU (e.g. "Apple M2 Ultra").
    pub fn name(&self) -> String {
        // name() returns Retained<NSString>; NSString implements Display.
        self.inner.name().to_string()
    }

    /// Raw borrow of the underlying `MTLDevice` protocol object, for use by
    /// sibling modules in this crate (e.g. `bridge`).
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.inner
    }
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice")
            .field("name", &self.name())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn system_default_device_acquires_and_has_a_name() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let name = device.name();
        assert!(!name.is_empty(), "device name should be non-empty");
        eprintln!("Metal device: {name}");
    }

    #[test]
    fn clone_is_cheap_and_gives_same_name() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let clone = device.clone();
        assert_eq!(device.name(), clone.name());
    }

    #[test]
    fn debug_includes_name() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let debug_str = format!("{device:?}");
        assert!(
            debug_str.contains("MetalDevice"),
            "debug should contain type name"
        );
        assert!(
            debug_str.contains("name"),
            "debug should contain field name"
        );
    }
}
