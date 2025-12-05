import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Bell, User } from "lucide-react"
import { ThemeToggle } from "@/components/ThemeToggle"

export function Header() {
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-4 border-b bg-background px-6">
      <div className="flex flex-1 items-center gap-4">
        <form className="w-full max-w-[400px]">
          <div className="relative">
            <Input 
              type="search" 
              placeholder="Search experiments..." 
              className="w-full md:w-[300px] lg:w-[400px]"
            />
          </div>
        </form>
      </div>
      <div className="flex items-center gap-2">
        <ThemeToggle />
        <Button variant="ghost" size="icon">
          <Bell className="h-4 w-4" />
          <span className="sr-only">Notifications</span>
        </Button>
        <Button variant="ghost" size="icon">
          <User className="h-4 w-4" />
          <span className="sr-only">User menu</span>
        </Button>
      </div>
    </header>
  )
}
