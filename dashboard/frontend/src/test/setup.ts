import React from "react";
import { vi } from "vitest";

import "@testing-library/jest-dom/vitest";

vi.mock("echarts-for-react", () => ({
  default: () => React.createElement("div", { "data-testid": "chart" }),
}));

vi.mock("framer-motion", async () => {
  const React = await import("react");

  return {
    motion: new Proxy(
      {},
      {
        get: (_target, key: string) => {
          return ({ children, ...props }: { children?: React.ReactNode; [value: string]: unknown }) => {
            const elementProps = { ...props };

            delete elementProps.initial;
            delete elementProps.animate;
            delete elementProps.transition;
            delete elementProps.whileHover;
            delete elementProps.whileTap;
            delete elementProps.exit;
            delete elementProps.layout;

            return React.createElement(key, elementProps, children);
          };
        },
      },
    ),
  };
});
